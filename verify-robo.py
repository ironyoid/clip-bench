import json
import os
import pickle

import cv2
import numpy as np


DATASET_PATH = "dataset/robotics_kitchen_dataset_v3"
ANN_PATH = f"{DATASET_PATH}/annotations/ground_truth/robotics_kitchen.json"


def decode_uncompressed_rle(rle):
    h, w = rle["size"]
    counts = rle["counts"]
    flat = np.zeros(h * w, dtype=np.uint8)
    idx = 0
    value = 0
    for count in counts:
        if value == 1:
            flat[idx:idx + count] = 1
        idx += count
        value = 1 - value
    return flat.reshape((h, w), order="F")


def load_robo_test(ann_path, coco_root):
    data = json.load(open(ann_path, "r", encoding="utf-8"))
    object_caption_path = os.path.join(coco_root, "objects_caption.json")
    masked_objects_dir = os.path.join(coco_root, "masked_objects")
    object_captions = json.load(
        open(object_caption_path, "r", encoding="utf-8"))
    os.makedirs(masked_objects_dir, exist_ok=True)

    images = []
    image_ids = []
    captions = []
    caption_ids = []

    annotations = data["annotations"]
    for frame_id in sorted(annotations.keys(), key=lambda x: int(x)):
        for mask_info in annotations[frame_id]["masks"]:
            mask_rel_path = mask_info["mask_path"]
            mask_name = os.path.basename(mask_rel_path)
            parts = mask_name.replace(".pkl", "").split("_")
            frame_num = int(parts[0].replace("frame", ""))
            object_id = int(mask_info["object_id"])
            out_path = os.path.join(
                masked_objects_dir, f"frame{frame_num}_obj{object_id}.jpg"
            )

            if not os.path.exists(out_path):
                frame_path = os.path.join(
                    coco_root, "frames", "robotics_kitchen", f"{frame_num - 1}.jpg"
                )
                mask_path = os.path.join(
                    coco_root, "mask_cache", mask_rel_path)

                image = cv2.imread(frame_path)
                rle = pickle.load(open(mask_path, "rb"))
                mask = decode_uncompressed_rle(rle).astype(bool)

                masked = image.copy()
                masked[~mask] = 0
                ys, xs = np.where(mask)
                y1, y2 = ys.min(), ys.max() + 1
                x1, x2 = xs.min(), xs.max() + 1
                object_crop = masked[y1:y2, x1:x2]
                cv2.imwrite(out_path, object_crop)

            images.append(out_path)
            image_ids.append(object_id)

    present_object_ids = sorted(set(image_ids))
    for object_id in present_object_ids:
        captions.append(object_captions[str(object_id)]["object_caption"].strip())
        caption_ids.append(object_id)

    return images, image_ids, captions, caption_ids


def show_samples_with_opencv(images, image_ids, captions, caption_ids):
    caption_by_object_id = {caption_ids[i]: captions[i] for i in range(len(caption_ids))}
    for i in range(len(images)):
        image = cv2.imread(images[i])
        object_id = image_ids[i]
        caption = caption_by_object_id[object_id]
        lines = [
            f"sample: {i + 1}/{len(images)}",
            f"object_id(image_id): {object_id}",
            f"caption_id: {object_id}",
            f"caption: {caption}",
            "key: any next | q/esc quit",
        ]

        panel_w = max(image.shape[1], 1200)
        panel_h = image.shape[0] + 170
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        panel[170:170 + image.shape[0], :image.shape[1]] = image

        y = 30
        for line in lines:
            cv2.putText(
                panel,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            y += 30

        cv2.imshow("robotics_kitchen_dataset_v3_verify", panel)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


def main():
    images, image_ids, captions, caption_ids = load_robo_test(
        ANN_PATH, DATASET_PATH)
    print(f"Masked object images: {len(images)}")
    print(f"Captions: {len(captions)}")
    show_samples_with_opencv(images, image_ids, captions, caption_ids)


if __name__ == "__main__":
    main()
