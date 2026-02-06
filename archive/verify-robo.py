import cv2
import numpy as np
from parsers import load_robo_dataset


DATASET_PATH = "dataset/robotics_kitchen_dataset_v3"
ANN_PATH = f"{DATASET_PATH}/annotations/ground_truth/robotics_kitchen.json"


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
    images, image_ids, captions, caption_ids = load_robo_dataset(
        ANN_PATH, DATASET_PATH)
    print(f"Masked object images: {len(images)}")
    print(f"Captions: {len(captions)}")
    show_samples_with_opencv(images, image_ids, captions, caption_ids)


if __name__ == "__main__":
    main()
