"""Run the MLLM judge over extracted frames.

Reads the frame folders produced by `generate/extract_frames.py`, asks the three
questions in `criteria.py` about each video, and writes one JSON line per
judgment. The judge is never shown the sentence -- only the verb, the object and
the target state -- so the same question is asked of every condition of an item.

The backend is any OpenAI-compatible chat endpoint, which covers hosted APIs and
a locally served model (vLLM, SGLang, llama.cpp) without a second code path.

    export JUDGE_API_KEY=...  JUDGE_BASE_URL=https://...   # or a local server
    python judge.py --frames /data/frames/wan2.2-ti2v-5b-480p --model gpt-5.2

    # judge reliability: the same videos a second time, to separate judge noise
    # from video-model instability
    python judge.py --frames ... --model ... --repeat 2
"""

import argparse
import base64
import json
import os
import time
from pathlib import Path

from criteria import build_prompt

ANSWERS = {"yes", "no", "unclear"}
QUESTION_IDS = ("action", "initial", "final")


REQUIRED_FIELDS = ("verb_base", "other_verb_base", "noun")


def load_items(stimuli_path):
    items = {r["item_id"]: r for r in
             (json.loads(l) for l in stimuli_path.read_text().splitlines() if l)}
    if not items:
        raise SystemExit(f"{stimuli_path} is empty")
    # The judge is shown the bare verb, never the gerund and never the sentence,
    # so these fields have to be present. They were added after the first
    # stimuli files were built, and a copy taken before that fails here rather
    # than mid-run.
    missing = [f for f in REQUIRED_FIELDS if f not in next(iter(items.values()))]
    if missing:
        raise SystemExit(
            f"{stimuli_path} predates fields {missing}.\n"
            f"  Rebuild it with the current code and copy it across again:\n"
            f"    cd probe && python build_stimuli.py --n-generate 40\n"
            f"  Both machines must use the SAME file -- item ids shift between "
            f"builds, and a mismatch silently judges the wrong video."
        )
    return items


def find_videos(frames_root, seeds=None, conditions=None):
    """Frame folders laid out as <root>/item0007/prog__seed42/frame_001.jpg."""
    found = []
    for folder in sorted(frames_root.glob("item*/*")):
        if not folder.is_dir():
            continue
        condition, _, seed = folder.name.partition("__seed")
        if not seed.isdigit():
            continue
        seed = int(seed)
        if seeds and seed not in seeds:
            continue
        if conditions and condition not in conditions:
            continue
        found.append({
            "item_id": int(folder.parent.name.removeprefix("item")),
            "condition": condition, "seed": seed, "frames": folder,
        })
    return found


def encode(path):
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def parse_answer(raw):
    """Pull the three answers out of the model's JSON, rejecting anything that
    is not one of the three allowed values -- a judge that invents a fourth
    answer has not understood the task, and silently coercing it would hide
    that."""
    out = {}
    for qid in QUESTION_IDS:
        block = raw.get(qid)
        if not isinstance(block, dict):
            raise ValueError(f"missing block for {qid!r}")
        answer = str(block.get("answer", "")).strip().lower()
        if answer not in ANSWERS:
            raise ValueError(f"{qid}: bad answer {answer!r}")
        out[qid] = {"answer": answer, "evidence": str(block.get("evidence", ""))}
    return out


def judge_one(client, model, job, item, max_frames, max_tokens,
              known_negative=False):
    frames = sorted(job["frames"].glob("frame_*.jpg"))[:max_frames]
    if not frames:
        raise FileNotFoundError(f"no frames in {job['frames']}")

    # The known-negative probe asks the same video about the OTHER verb's target
    # state. A grating video has not been "cut into very fine pieces", so the
    # answer is no in advance -- for every video already on disk, which is what
    # makes specificity measurable without labelling anything.
    verb = item["other_verb_base"] if known_negative else item["verb_base"]
    prompt = build_prompt(verb, item["noun"], len(frames))

    content = [{"type": "text", "text": prompt}]
    content += [{"type": "image_url", "image_url": {"url": encode(f)}} for f in frames]

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"},
        max_completion_tokens=max_tokens,
    )
    text = resp.choices[0].message.content
    return parse_answer(json.loads(text)), len(frames), verb, resp


def done_keys(out_path, probe):
    """(item, condition, seed, pass) already judged under this probe."""
    done = set()
    if not out_path.exists():
        return done
    for line in out_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("status") == "ok" and r.get("probe", "target") == probe:
                done.add((r["item_id"], r["condition"], r["seed"], r["pass"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True,
                    help="a model directory produced by extract_frames.py")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="default <frames>/judgments.jsonl")
    ap.add_argument("--stimuli", type=Path,
                    default=Path(__file__).parent.parent / "probe" / "stimuli.jsonl")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--max-frames", type=int, default=20,
                    help="OSCBench samples 20 for MLLM evaluation")
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--repeat", type=int, default=1,
                    help="judge every video N times; pass 2 to measure "
                         "test-retest reliability, which is what separates "
                         "judge noise from video-model instability")
    ap.add_argument("--known-negative", action="store_true",
                    help="judge each video against the OTHER verb's target "
                         "state, whose answer is 'no' by construction. This is "
                         "how specificity is measured without any labels")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--base-url", default=os.environ.get("JUDGE_BASE_URL"))
    ap.add_argument("--dry-run", action="store_true",
                    help="print one built prompt and the job count, call nothing")
    args = ap.parse_args()

    items = load_items(args.stimuli)
    jobs = find_videos(args.frames, args.seeds, args.conditions)
    if not jobs:
        raise SystemExit(f"no frame folders under {args.frames}")
    jobs = [j for j in jobs if j["item_id"] in items]

    probe = "known_negative" if args.known_negative else "target"
    out_path = args.out or args.frames / "judgments.jsonl"
    already = done_keys(out_path, probe)
    todo = [dict(j, **{"pass": p}) for j in jobs for p in range(1, args.repeat + 1)
            if (j["item_id"], j["condition"], j["seed"], p) not in already]
    if args.limit:
        todo = todo[:args.limit]

    print(f"{len(jobs)} videos x {args.repeat} pass(es), "
          f"{len(already)} already judged, {len(todo)} to go")

    if args.dry_run:
        j = todo[0]
        item = items[j["item_id"]]
        n = len(list(j["frames"].glob("frame_*.jpg"))[:args.max_frames])
        print(f"\n--- prompt for item{j['item_id']:04d} {j['condition']} "
              f"({n} frames) ---")
        verb = item["other_verb_base"] if args.known_negative else item["verb_base"]
        print(build_prompt(verb, item["noun"], n))
        return

    from openai import OpenAI  # noqa: PLC0415

    client = OpenAI(api_key=os.environ.get("JUDGE_API_KEY"), base_url=args.base_url)

    # Token totals, so the cost of the full run is extrapolated from measured
    # usage rather than guessed. 20 frames per call is the dominant cost and it
    # is worth knowing the real number before committing to thousands of calls.
    tokens = {"prompt": 0, "completion": 0}
    t0 = time.time()
    with out_path.open("a") as fh:
        for n, job in enumerate(todo, 1):
            item = items[job["item_id"]]
            rec = {"item_id": job["item_id"], "condition": job["condition"],
                   "seed": job["seed"], "pass": job["pass"], "probe": probe,
                   "gerund": item["gerund"], "noun": item["noun"],
                   "aspectual_class": item["aspectual_class"],
                   "judge_model": args.model}
            try:
                answers, n_frames, verb, resp = judge_one(
                    client, args.model, job, item, args.max_frames,
                    args.max_tokens, args.known_negative)
                usage = getattr(resp, "usage", None)
                rec.update(status="ok", n_frames=n_frames, judged_verb=verb,
                           **answers)
                if usage is not None:
                    rec["usage"] = usage.model_dump()
                    tokens["prompt"] += getattr(usage, "prompt_tokens", 0) or 0
                    tokens["completion"] += getattr(usage, "completion_tokens", 0) or 0
            except Exception as exc:  # one bad video must not kill the batch
                rec.update(status="error", error=f"{type(exc).__name__}: {exc}")
                print(f"  ERROR item{job['item_id']:04d} {job['condition']}: "
                      f"{rec['error']}")
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if n % 20 == 0 or n == len(todo):
                rate = (time.time() - t0) / n
                print(f"  {n}/{len(todo)}  {rate:.1f}s/video  "
                      f"eta {(len(todo) - n) * rate / 60:.0f}min")

    done = max(len(todo), 1)
    print(f"done. judgments: {out_path}")
    if tokens["prompt"]:
        print(f"  tokens: {tokens['prompt']:,} prompt + "
              f"{tokens['completion']:,} completion")
        print(f"  per call: {tokens['prompt'] // done:,} prompt "
              f"({args.max_frames} frames)")
        print(f"  extrapolated to 9,000 calls: "
              f"{tokens['prompt'] // done * 9000 / 1e6:.0f}M prompt tokens")


if __name__ == "__main__":
    main()
