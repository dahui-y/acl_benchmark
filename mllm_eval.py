import re
import json
import base64
from pathlib import Path
import time

from openai import OpenAI


OPENAI_API_KEY = "Your OpenAI API key here"
MODEL_NAME = "gpt-5.2"
VIDEO_DURATION = 5.0
MAX_OUTPUT_TOKENS = 1200

ROOT_DIR = Path("/Users/hanxianjing/proj/Video benchmark/selected_frames")
OUTPUT_JSONL = Path("/Users/hanxianjing/proj/Video benchmark/gpt5.2_evidence.jsonl")

EVALUATION_CRITERIA = {
    "1a": {
        "id": "1a",
        "category": "OBJECT",
        "question": "Subject Alignment",
        "description": "Is the subject present and correct (i.e., the main actor, e.g., a person or a hand)? Please select NA if the prompt does not specify a subject.",
        "scoring": {
            1: "Subject is absent or replaced by something entirely unrelated.",
            2: "Subject is present but does not match the expected category.",
            3: "Subject is of the correct category but exhibits major attribute errors.",
            4: "Subject is correct and well-rendered, with only minor attribute errors.",
            5: "Subject perfectly matches the prompt in category, form, and attributes."
        }
    },
    "1b": {
        "id": "1b",
        "category": "OBJECT",
        "question": "Manipulated Object Alignment",
        "description": "Is the manipulated object present and correct (e.g., carrots or tomatoes)?",
        "scoring": {
            1: "Manipulated object is absent, or a completely different object is present.",
            2: "Manipulated object is of the wrong category or is severely distorted.",
            3: "Manipulated object is of the correct category but shows major visual inaccuracies.",
            4: "Manipulated object is correct and realistic, with only minor visual inaccuracies.",
            5: "Manipulated object is realistic and provides a perfect visual match."
        }
    },
    "2a": {
        "id": "2a",
        "category": "ACTION",
        "question": "Action Accuracy",
        "description": "Does the performed action match the action in the prompt (e.g., slicing or roasting)?",
        "scoring": {
            1: "A fundamentally different action is performed.",
            2: "The intended action is recognizable but executed in a physically incorrect way.",
            3: "The correct action is performed but with clear physical or logical flaws.",
            4: "Action is performed correctly, but motion appears slightly unnatural.",
            5: "Action is executed in a physically plausible, natural manner."
        }
    },
    "3a": {
        "id": "3a",
        "category": "STATE CHANGE",
        "question": "Object State Change Accuracy",
        "description": "Is the object state change correct and as expected (e.g., an apple changing from whole to slices)?",
        "scoring": {
            1: "Object state change is illogical or unrelated to the action.",
            2: "Object state change does not match the expected outcome.",
            3: "Object state change is partially correct, but major inaccuracies remain.",
            4: "Object state change is generally correct, with minor issues.",
            5: "Object state change is accurate and matches the expected outcome exactly."
        }
    },
    "3b": {
        "id": "3b",
        "category": "STATE CHANGE",
        "question": "Object Change Continuity & Consistency",
        "description": "Is the object state change continuous and natural, without any unnatural object appearances or disappearances (e.g., the object suddenly popping into the scene)?",
        "scoring": {
            1: "State change is highly discontinuous, with obvious jumps or objects suddenly appearing/disappearing.",
            2: "State change is discontinuous or has noticeable object appearances/disappearances.",
            3: "State change is mostly continuous but includes small jumps or object inconsistencies.",
            4: "State change is continuous and natural, with only minimal, non-disruptive inconsistencies.",
            5: "State change is smooth and continuous, with no unnatural object appearances/disappearances."
        }
    },
    "4a": {
        "id": "4a",
        "category": "SCENE",
        "question": "Scene Alignment",
        "description": "Does the background and environment match the prompt (e.g., a kitchen or a market) Please select NA if the prompt does not specify a scene.?",
        "scoring": {
            1: "Scene directly contradicts the prompt.",
            2: "Scene is generic or ambiguous and lacks required details.",
            3: "Scene partially matches the prompt but contains notable attribute inaccuracies.",
            4: "Scene contains correct elements with only minor attribute inaccuracies.",
            5: "Scene is a detailed and accurate match to the prompt's setting."
        }
    },
    "5a": {
        "id": "5a",
        "category": "GENERAL QUALITY",
        "question": "Realism",
        "description": "Does this video look like a real-world video?",
        "scoring": {
            1: "Video looks artificial, distorted, or obviously fake.",
            2: "Many visual artifacts; motion, lighting, or textures do not resemble real footage.",
            3: "Some elements look real, but noticeable artifacts reduce overall realism.",
            4: "Video appears close to real with only minor visual imperfections.",
            5: "Video looks convincingly real with natural motion, lighting, and textures."
        }
    },
    "5b": {
        "id": "5b",
        "category": "GENERAL QUALITY",
        "question": "Aesthetic",
        "description": "Is the video visually appealing? Are the colors harmonious and is the content rich?",
        "scoring": {
            1: "Video is visually unappealing, with distracting colors or dull/empty content.",
            2: "Some attempt at aesthetics, but colors clash or the content feels sparse.",
            3: "Overall visually fine, with moderate harmony and adequate content richness.",
            4: "Visually appealing, with harmonious colors and rich, engaging content.",
            5: "Highly pleasing visuals, strong color harmony, and rich, well-composed content throughout."
        }
    }
}


def criteria_to_text() -> str:
    lines = []
    for k in ["1a","1b","2a","3a","3b","4a","5a","5b"]:
        c = EVALUATION_CRITERIA[k]
        lines.append(f"{k}: {c['question']}")
        lines.append(f"Description: {c['description']}")
        lines.append("Scoring:")
        for s, d in c["scoring"].items():
            lines.append(f"  {s}: {d}")
        lines.append("")
    return "\n".join(lines)



def build_prompt(video_prompt: str, n_frames: int) -> str:
    return f"""Suppose you are an expert in judging and evaluating the quality of AI-generated videos.

You are given {n_frames} frames evenly sampled from a {VIDEO_DURATION}-second AI-generated video.

AI-generated videos may exhibit anomalies such as unnatural object appearance or disappearance, physically implausible state changes, and temporal inconsistencies across frames. They may also contain visual artifacts or unnatural textures.

Video Prompt:
"{video_prompt}"

YOUR TASK:
Analyze these frames chronologically and evaluate the video using the following criteria.

{criteria_to_text()}

Instructions:
- Evaluate each criterion INDEPENDENTLY.
- For each criterion, first identify the relevant factual evidence from the frames relevant to that criterion.
- Then assign a score based on that evidence.

Output Format:
Return the result strictly in JSON format.
{{
  "1a": {{"evidence": "...", "score": [1-5]}},
  "1b": {{"evidence": "...", "score": [1-5]}},
  "2a": {{"evidence": "...", "score": [1-5]}},
  "3a": {{"evidence": "...", "score": [1-5]}},
  "3b": {{"evidence": "...", "score": [1-5]}},
  "4a": {{"evidence": "...", "score": [1-5]}},
  "5a": {{"evidence": "...", "score": [1-5]}},
  "5b": {{"evidence": "...", "score": [1-5]}}
}}
"""

def load_prompt_from_folder(folder: Path) -> str:
    name = folder.name
    name = re.sub(r"^\d+_", "", name)
    name = re.sub(r"_[a-zA-Z]{2}$", "", name)
    return name.replace("_", " ").strip()


def encode_image(p: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def evaluate_one_video(client: OpenAI, folder: Path) -> dict:
    t0 = time.perf_counter()

    frames = sorted(folder.glob("frame_*.jpg"))

    video_prompt = load_prompt_from_folder(folder)
    prompt_text = build_prompt(video_prompt, len(frames))

    content = [{"type": "input_text", "text": prompt_text}]
    for fp in frames:
        content.append({"type": "input_image", "image_url": encode_image(fp)})

    resp = client.responses.create(
        model=MODEL_NAME,
        input=[{"role": "user", "content": content}],
        text={"format": {"type": "json_object"}},
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    t1 = time.perf_counter()
    elapsed = t1 - t0

    result = json.loads(resp.output_text)
    usage = resp.usage.model_dump() if resp.usage else None

    return {
        "video_folder": folder.name,
        "prompt": video_prompt,
        "result": result,
        "usage": usage,
        "elapsed_sec": round(elapsed, 3),
    }

def main():
    client = OpenAI(api_key=OPENAI_API_KEY)

    folders = sorted([p for p in ROOT_DIR.iterdir() if p.is_dir()])
    print(f"Found {len(folders)} videos")

    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for i, folder in enumerate(folders, 1):
            print(f"[{i}/{len(folders)}] {folder.name}")
            data = evaluate_one_video(client, folder)
            print("Token usage:", data["usage"])
            print(f"[{i}/{len(folders)}] {folder.name} | time: {data['elapsed_sec']}s | usage: {data['usage']}")
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"\nDone. Results saved to {OUTPUT_JSONL.resolve()}")


if __name__ == "__main__":
    main()