"""Run a detector over the calibration cells and store every detection.

The output is per-instance, not per-count: every box above a permissive floor is
written with its score, and the count is formed later in `report.py`. That split
matters. GenEval reports raising its confidence threshold from 0.3 to 0.9 for
the counting task specifically, because same-class instances draw a spray of
low-confidence boxes. Storing raw detections lets the threshold be swept over
the same predictions instead of taken on trust -- if 0.9 is right for our object
distribution it will show up as a maximum, and if it is not, we will see that
before building anything on top of it.

Two detectors, on purpose:

  mask2former  -- the closed-vocabulary COCO model GenEval itself uses, so the
                  number is comparable to a published one
  owlv2        -- open-vocabulary, which is what the real suite needs, since
                  the object list will not stay inside COCO's 80 classes

Agreement between them on our own generated images is the substitute for the
human check we are not doing, so both have to be calibrated here first.

    python detect.py --detector mask2former
    python detect.py --detector owlv2
"""

import argparse
import json
import os
import time
from pathlib import Path
from urllib.request import urlopen

COCO_URL = "http://images.cocodataset.org/val2017/{}"

MODELS = {
    "mask2former": "facebook/mask2former-swin-large-coco-instance",
    "mask2former-small": "facebook/mask2former-swin-small-coco-instance",
    "owlv2": "google/owlv2-base-patch16-ensemble",
    "owlv2-large": "google/owlv2-large-patch14-ensemble",
    # An MLLM asked for the number directly. Worth testing because the whole
    # design hinges on the verifier, only detectors have been tried, and unlike
    # the video judge this one can be calibrated -- COCO supplies the gold
    # counts that the video work never had.
    "mllm": "Qwen/Qwen2.5-VL-3B-Instruct",
    "mllm-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "mllm-api": None,   # --model-id names the hosted model
}

COUNT_PROMPT = """How many {noun} are visible in this image?

Count every separate {noun}, including ones that are partly hidden behind \
something else or cut off at the edge of the frame. Do not count anything that \
is not a {noun}.

Answer with this JSON and nothing else:
{{"count": <integer>}}"""

# A permissive floor. Anything above it is stored; the operating threshold is
# chosen in report.py. Do not raise this -- it would silently truncate the
# sweep's lower half and make whatever threshold wins look better than it is.
STORE_FLOOR = 0.05


def fetch(cell, cache):
    """Cells carry their own URL: LVIS spans both COCO splits, so the val2017
    path cannot be assumed."""
    path = cache / cell["file_name"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = cell.get("url") or COCO_URL.format(cell["file_name"])
        with urlopen(url, timeout=60) as r:
            path.write_bytes(r.read())
    return path


# This checkpoint labels six classes with their older VOC names, so a plain
# string comparison against the COCO category silently returns zero detections
# for all six -- which reads as "the detector cannot find a sofa" rather than as
# a naming mismatch. Derived by diffing the model's id2label against the COCO
# categories; the assertion below keeps it honest if a checkpoint changes.
COCO_TO_MODEL_LABEL = {
    "airplane": "aeroplane", "couch": "sofa", "dining table": "diningtable",
    "motorcycle": "motorbike", "potted plant": "pottedplant", "tv": "tvmonitor",
}


class Mask2Former:
    """Closed-vocabulary instance segmentation over COCO's 80 classes."""

    def __init__(self, model_id, device):
        import torch  # noqa: PLC0415
        from transformers import (AutoImageProcessor,  # noqa: PLC0415
                                  Mask2FormerForUniversalSegmentation)
        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_id).to(device).eval()
        self.device = device
        self.label = self.model.config.id2label

    def name_for(self, class_name):
        """The label this checkpoint uses for a COCO category, verified."""
        name = COCO_TO_MODEL_LABEL.get(class_name, class_name)
        if name not in self.label.values():
            raise KeyError(
                f"{class_name!r} maps to {name!r}, which this checkpoint does "
                f"not know. Its labels are: {sorted(self.label.values())}")
        return name

    def scores_for(self, image, class_name):
        """All detection scores for `class_name` in this image."""
        target = self.name_for(class_name)
        inputs = self.proc(images=image, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        res = self.proc.post_process_instance_segmentation(
            out, target_sizes=[image.size[::-1]], threshold=STORE_FLOOR)[0]
        return ([s["score"] for s in res["segments_info"]
                 if self.label[s["label_id"]] == target], None)


class OwlV2:
    """Open-vocabulary detection: the class name is the query."""

    def __init__(self, model_id, device):
        import torch  # noqa: PLC0415
        from transformers import (Owlv2ForObjectDetection,  # noqa: PLC0415
                                  Owlv2Processor)
        self.torch = torch
        self.proc = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            model_id).to(device).eval()
        self.device = device

    def scores_for(self, image, class_name):
        # "a photo of a X" is the phrasing OWL-ViT was trained to match; a bare
        # noun measurably underperforms it.
        query = [[f"a photo of a {class_name}"]]
        inputs = self.proc(text=query, images=image,
                           return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        res = self.proc.post_process_grounded_object_detection(
            out, target_sizes=self.torch.tensor([image.size[::-1]]),
            threshold=STORE_FLOOR)[0]
        # Boxes come back overlapping -- a detector returning six boxes on one
        # apple would report six apples. Mask2Former needs no equivalent step
        # because its instance post-processing assigns each pixel to at most one
        # segment, so its instances cannot double-count by construction. Boxes
        # are stored so the IoU can be revisited without re-running anything.
        keep = self.nms(res["boxes"], res["scores"], iou=0.5)
        return ([float(res["scores"][i]) for i in keep],
                [[round(float(v), 1) for v in res["boxes"][i]] for i in keep])

    def nms(self, boxes, scores, iou):
        from torchvision.ops import nms  # noqa: PLC0415
        return nms(boxes, scores, iou).tolist()


class MllmLocal:
    """Open-weights vision-language model, asked for the integer directly.

    Image tokens dominate the cost, so max_pixels is capped. Counting needs
    enough resolution to separate touching instances, and this is the knob that
    trades that against runtime -- it is exposed rather than fixed because the
    right value is an empirical question, the same way the detectors' confidence
    threshold was.
    """

    def __init__(self, model_id, device, max_pixels=768 * 28 * 28):
        import torch  # noqa: PLC0415
        from transformers import (AutoModelForImageTextToText,  # noqa: PLC0415
                                  AutoProcessor)
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(model_id, max_pixels=max_pixels)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16).to(device).eval()
        self.device = device

    def count_for(self, image, class_name):
        msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": COUNT_PROMPT.format(noun=class_name)}]}]
        text = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=[image],
                           return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=24,
                                      do_sample=False)
        reply = self.proc.batch_decode(
            out[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True)[0]
        return parse_count(reply), reply


class MllmApi:
    """Any OpenAI-compatible chat endpoint, so a hosted model can be calibrated
    on exactly the same cells as the local one."""

    def __init__(self, model_id, device, base_url=None, api_key=None):
        from openai import OpenAI  # noqa: PLC0415
        self.client = OpenAI(api_key=api_key or os.environ.get("JUDGE_API_KEY"),
                             base_url=base_url or os.environ.get("JUDGE_BASE_URL"))
        self.model_id = model_id

    def count_for(self, image, class_name):
        import base64  # noqa: PLC0415
        import io  # noqa: PLC0415
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=92)
        url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": COUNT_PROMPT.format(noun=class_name)},
                {"type": "image_url", "image_url": {"url": url}}]}],
            response_format={"type": "json_object"}, max_completion_tokens=64)
        reply = resp.choices[0].message.content
        return parse_count(reply), reply


def parse_count(reply):
    """Pull the integer out, refusing to guess.

    A reply that does not contain a number is an error, not a zero. Coercing it
    would put the model's confusion into the count column, where it would look
    like the model confidently saw nothing.
    """
    import re  # noqa: PLC0415
    m = re.search(r'"count"\s*:\s*(-?\d+)', reply)
    if not m:
        m = re.search(r"(-?\d+)", reply)
    if not m:
        raise ValueError(f"no integer in reply: {reply!r}")
    n = int(m.group(1))
    if n < 0:
        raise ValueError(f"negative count in reply: {reply!r}")
    return n


def load_detector(name, device, model_id=None, max_pixels=None):
    model_id = model_id or MODELS[name]
    if name == "mllm-api":
        return MllmApi(model_id, device)
    if name.startswith("mllm"):
        kw = {"max_pixels": max_pixels} if max_pixels else {}
        return MllmLocal(model_id, device, **kw)
    cls = OwlV2 if name.startswith("owlv2") else Mask2Former
    return cls(model_id, device)


def done_keys(out_path):
    if not out_path.exists():
        return set()
    return {(r["image_id"], r["class"]) for r in
            map(json.loads, out_path.read_text().splitlines()) if r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="mask2former", choices=list(MODELS))
    ap.add_argument("--cells", type=Path,
                    default=Path(__file__).parent / "data" / "cells.jsonl")
    ap.add_argument("--cache", type=Path,
                    default=Path(__file__).parent / "data" / "images")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default=None,
                    help="default cuda when available, else cpu")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model-id", default=None,
                    help="override the checkpoint / hosted model name")
    ap.add_argument("--max-pixels", type=int, default=None,
                    help="mllm only: image-token budget, as pixels")
    args = ap.parse_args()

    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_path = args.out or (args.cells.parent / f"pred_{args.detector}.jsonl")

    cells = [json.loads(l) for l in args.cells.read_text().splitlines() if l]
    already = done_keys(out_path)
    todo = [c for c in cells if (c["image_id"], c["class"]) not in already]
    # Count what is done BEFORE --limit trims the plan. Deriving it from the
    # trimmed list reports `--limit 4` on an empty output file as "411 done".
    n_done = len(cells) - len(todo)
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(cells)} cells, {n_done} done, {len(todo)} to go"
          f"  [{args.detector} on {device}]")
    if not todo:
        return

    det = load_detector(args.detector, device, args.model_id, args.max_pixels)
    t0 = time.time()
    with out_path.open("a") as fh:
        for n, cell in enumerate(todo, 1):
            rec = dict(cell, detector=args.detector)
            try:
                image = Image.open(fetch(cell, args.cache)).convert("RGB")
                if hasattr(det, "count_for"):
                    # An MLLM emits the number itself, so there is no threshold
                    # to sweep. report.py prefers `count` when it is present.
                    rec["count"], rec["reply"] = det.count_for(
                        image, cell["class"])
                    rec["scores"] = [1.0] * rec["count"]
                else:
                    scores, boxes = det.scores_for(image, cell["class"])
                    order = sorted(range(len(scores)), key=lambda i: -scores[i])
                    rec["scores"] = [scores[i] for i in order]
                    if boxes is not None:
                        rec["boxes"] = [boxes[i] for i in order]
                rec["status"] = "ok"
            except Exception as exc:  # one bad image must not kill the batch
                rec.update(status="error", error=f"{type(exc).__name__}: {exc}")
                print(f"  ERROR {cell['file_name']} {cell['class']}: {exc}")
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if n % 10 == 0 or n == len(todo):
                rate = (time.time() - t0) / n
                print(f"  {n}/{len(todo)}  {rate:.2f}s/img  "
                      f"eta {(len(todo) - n) * rate / 60:.0f}min", flush=True)
    print(f"done. {out_path}")


if __name__ == "__main__":
    main()
