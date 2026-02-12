import os

import cv2
import numpy as np
from PIL import Image

SAVE_DIR = "viz_output"


def visualize_topk(captions, images, ranked_indices, caption_match_ids, image_ids, model_name="model", k=10, tile_size=160, cols=5):
    rows = (k + cols - 1) // cols

    for i, caption in enumerate(captions):
        top = ranked_indices[i][:k]
        query_obj = caption_match_ids[i]

        grid_w = cols * tile_size

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"[{i+1}/{len(captions)}] {caption}"
        header_h = 32
        header = np.zeros((header_h, grid_w, 3), dtype=np.uint8)
        cv2.putText(header, text, (8, 22), font, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

        grid_h = rows * tile_size
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        for j, idx in enumerate(top):
            img = Image.open(images[idx]).convert("RGB").resize(
                (tile_size, tile_size), Image.BICUBIC)
            tile = np.array(img)[:, :, ::-1]
            r, c = j // cols, j % cols
            y, x = r * tile_size, c * tile_size
            grid[y:y + tile_size, x:x + tile_size] = tile
            if image_ids[idx] == query_obj:
                cv2.rectangle(grid, (x + 1, y + 1), (x + tile_size -
                              2, y + tile_size - 2), (0, 255, 0), 3)

        canvas = np.vstack([header, grid])

        cv2.imshow("topk", canvas)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('p'):
                os.makedirs(SAVE_DIR, exist_ok=True)
                path = os.path.join(
                    SAVE_DIR, f"{i+1:04d}_{model_name}.png")
                cv2.imwrite(path, canvas)
                print(f"Saved: {path}")
                continue
            break
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
