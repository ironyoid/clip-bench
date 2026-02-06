import json
import ollama
from tqdm import tqdm

INPUT_PATH = "../dataset/robotics_kitchen_dataset_v3/objects_caption.json"
OUTPUT_PATH = "../dataset/robotics_kitchen_dataset_v3/objects_caption_prephrases.json"

MODEL = "gemma3:12b"
# MODEL = "gemma3:4b"
# MODEL = "phi3:mini"

N = 4
TEMPERATURE = 2

SYSTEM_PROMPT = """You are a helpful prephrase generator for image retrieval.

Rules:
- Generate short prephrases that can be prepended to the caption.
- Keep the meaning. Do not add new details or remove important ones.
- Preserve numbers, colors, attributes, and relations.
- Do not mention "caption" or "prephrase" or add commentary.
- Output must be valid JSON only.
- Use natural, varied wording.
- Each prephrase should be under 20 words.
- You are not allowed to produce less than 4 prephrases.

Return exactly N prephrases as:
{"prephrases":["...","..."]}
"""

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

out = {}
for obj_id, obj in tqdm(list(data.items()), desc="Objects"):
    caption = obj.get("object_caption", "").strip()
    payload = None
    for _ in range(3):
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"CAPTION: {caption}\nN: {N}"},
            ],
            format="json",
            options={"temperature": TEMPERATURE},
        )
        try:
            payload = json.loads(response["message"]["content"])
            if isinstance(payload, dict) and len(payload.get("prephrases", [])) == N:
                break
            payload = None
        except json.JSONDecodeError:
            payload = None
    if payload is None:
        raise ValueError(
            f"Model did not return valid JSON. object_id={obj_id} caption={caption}"
        )
    out[obj_id] = {
        **obj,
        "prephrases": payload["prephrases"],
    }

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f"Saved {len(out)} objects to {OUTPUT_PATH}")
