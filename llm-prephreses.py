import json
import os
import ollama
from tqdm import tqdm

ANN_PATH = "dataset/coco2014/annotations/karpathy_test.json"
COCO_ROOT = "dataset/coco2014"
OUTPUT_PATH = "dataset/coco2014/annotations/karpathy_paraphrases_too_hot.json"
# MODEL = "gemma3:12b"
MODEL = "gemma3:4b"
# MODEL = "phi3:mini"

N = 4
TEMPERATURE = 2

SYSTEM_PROMPT = """You are a helpful caption paraphraser for image retrieval.

Rules:
- Keep the meaning. Do not add new details or remove important ones.
- Preserve numbers, colors, attributes, and relations.
- Do not mention \"caption\" or \"paraphrase\" or add commentary.
- Output must be valid JSON only.
- You can use synonyms.
- Use natural, varied wording and structure.
- Each under 25 words.
- You are not allowed to produce less than 4 paraphrases.

Return exactly N paraphrases as:
{"paraphrases":["...","..."]}
"""
# print(SYSTEM_PROMPT)

with open(ANN_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

last_id = None
existing_count = 0
content = ""
last_valid_end = 0
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    decoder = json.JSONDecoder()
    i = content.find("[")
    if i != -1:
        i += 1
        last_valid_end = i
        while True:
            while i < len(content) and content[i] in " \n\r\t,":
                i += 1
            if i >= len(content) or content[i] == "]":
                last_valid_end = i
                break
            try:
                obj, j = decoder.raw_decode(content, i)
            except json.JSONDecodeError:
                break
            existing_count += 1
            if isinstance(obj, dict) and "image_id" in obj:
                last_id = obj["image_id"]
            i = j
            last_valid_end = i

images = []
found_last = last_id is None
for img in data["images"]:
    if img["split"] != "test":
        continue
    if not found_last:
        if img["cocoid"] == last_id:
            found_last = True
        continue
    images.append(img)

if existing_count:
    trimmed = content[:last_valid_end].rstrip()
    while trimmed.endswith(","):
        trimmed = trimmed[:-1].rstrip()
    if not trimmed.startswith("["):
        existing_count = 0
    else:
        f = open(OUTPUT_PATH, "w", encoding="utf-8")
        f.write(trimmed)
if not existing_count:
    f = open(OUTPUT_PATH, "w", encoding="utf-8")
    f.write("[\n")

need_comma = existing_count > 0
for img in tqdm(images, desc="Images"):
    image_path = os.path.join(COCO_ROOT, img["filepath"], img["filename"])
    out_captions = []
    for sent in img["sentences"]:
        caption = sent["raw"].strip()
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
                break
            except json.JSONDecodeError:
                payload = None
        if payload is None:
            raise ValueError(
                f"Model did not return valid JSON. image_id={img['cocoid']} "
                f"caption_id={sent['sentid']} caption={caption}"
            )
        out_captions.append(
            {
                "caption_id": sent["sentid"],
                "caption": caption,
                "paraphrases": payload["paraphrases"],
            }
        )
    if need_comma:
        f.write(",\n")
    json.dump(
        {
            "image_id": img["cocoid"],
            "image": image_path,
            "captions": out_captions,
        },
        f,
        ensure_ascii=False,
    )
    need_comma = True
f.write("\n]\n")
f.close()

print(f"Saved {len(images)} images to {OUTPUT_PATH}")
