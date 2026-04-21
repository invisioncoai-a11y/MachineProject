import cv2
import numpy as np

from DAL.preparation.config import IMG_SIZE, MAX_PATCHES


def extract_lesion_candidates(
    image_bgr,
    out_size=IMG_SIZE,
    max_patches=MAX_PATCHES,
    min_area_ratio=0.002
):
    h0, w0 = image_bgr.shape[:2]
    img = image_bgr.copy()

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    green_mask = cv2.inRange(hsv, np.array([25, 20, 20]), np.array([100, 255, 255]))
    brown_mask = cv2.inRange(hsv, np.array([5, 40, 20]), np.array([25, 255, 220]))
    yellow_mask = cv2.inRange(hsv, np.array([15, 20, 80]), np.array([40, 255, 255]))

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, dark_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    lesion_mask = cv2.bitwise_or(brown_mask, yellow_mask)
    lesion_mask = cv2.bitwise_or(lesion_mask, dark_mask)
    lesion_mask = cv2.bitwise_and(lesion_mask, cv2.bitwise_not(green_mask))

    k1 = np.ones((3, 3), np.uint8)
    k2 = np.ones((5, 5), np.uint8)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, k1)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, k2)
    lesion_mask = cv2.dilate(lesion_mask, k1, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(lesion_mask, connectivity=8)

    min_area = int(h0 * w0 * min_area_ratio)
    candidates = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        candidates.append((float(area), x, y, w, h))

    candidates = sorted(candidates, key=lambda z: z[0], reverse=True)[:max_patches]

    patches = []
    boxes = []

    if len(candidates) == 0:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        patch = cv2.resize(rgb, (out_size, out_size)).astype(np.float32) / 255.0
        patches.append(patch)
        boxes.append((0, 0, w0, h0))
        return patches, boxes, lesion_mask

    for _, x, y, w, h in candidates:
        pad_x = int(0.08 * w)
        pad_y = int(0.08 * h)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w0, x + w + pad_x)
        y2 = min(h0, y + h + pad_y)

        crop = image_bgr[y1:y2, x1:x2]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (out_size, out_size)).astype(np.float32) / 255.0

        patches.append(crop)
        boxes.append((x1, y1, x2, y2))

    return patches, boxes, lesion_mask