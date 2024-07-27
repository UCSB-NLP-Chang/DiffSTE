import os
import argparse

import cv2
import numpy as np
from matplotlib import pyplot as plt

from doctr.io import DocumentFile  # type: ignore
from doctr.models import detection_predictor  # type: ignore


def plot(image, si=[12, 12]):
    fig, ax = plt.subplots(figsize=si)
    ax.imshow(image, cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()


def crop(img, cxcy, crop_size=256):
    H, W, _ = img.shape
    cx, cy = cxcy
    half_crop_size = crop_size // 2

    # Calculate the top-left and bottom-right corners of the crop
    x1 = max(cx - half_crop_size, 0)
    y1 = max(cy - half_crop_size, 0)
    x2 = min(cx + half_crop_size, W)
    y2 = min(cy + half_crop_size, H)

    # Adjust the corners if the crop size is smaller than 255x255 pixels
    if x2 - x1 < crop_size:
        if x1 == 0:
            x2 = min(crop_size, W)
        else:
            x1 = max(x2 - crop_size, 0)

    if y2 - y1 < crop_size:
        if y1 == 0:
            y2 = min(crop_size, H)
        else:
            y1 = max(y2 - crop_size, 0)

    # Crop the image
    cropped_image = img[y1:y2, x1:x2]

    # If the crop is smaller than 255x255, pad it
    if cropped_image.shape[0] < crop_size or cropped_image.shape[1] < crop_size:
        padded_image = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        padded_image[:cropped_image.shape[0], :cropped_image.shape[1]] = cropped_image
        cropped_image = padded_image
    
    return cropped_image


def main(args):
    
    det_predictor = detection_predictor(
        arch=args.arch, 
        pretrained=True,
        pretrained_backbone=True,
        batch_size=1,
        assume_straight_pages=False,
        preserve_aspect_ratio=False,
        symmetric_pad=True
    )
    
    if args.device == "cuda":
        det_predictor = det_predictor.cuda().half()

    det_predictor.model.postprocessor.bin_thresh = args.bin_thresh
    
    input = DocumentFile.from_images(args.input_image)
    img = cv2.imread(args.input_image)
    H, W, _ = img.shape
    print(f"{H}, {W}")
    
    bboxes = det_predictor(input, return_maps=False)[0]['words']
    bboxes[:, :, 0] *= W
    bboxes[:, :, 1] *= H
    bboxes = bboxes.astype(np.int32)
    
    idx = np.random.randint(0, len(bboxes), size=1).item()
    bbox = bboxes[idx]
    cxcy = np.mean(bbox, axis=0, dtype=np.int32)
    
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.fillConvexPoly(mask, bbox, [255, 255, 255])
    
    imgc = crop(img, cxcy)
    maskc = crop(mask, cxcy)
    
    plot(mask)
    
    cv2.imwrite(os.path.join(args.output_dir, "sample.png"), imgc)
    cv2.imwrite(os.path.join(args.output_dir, "mask.png"), maskc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images for OCR using Doctr.')
    parser.add_argument('--input_image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output images')
    parser.add_argument('--arch', type=str, default='db_resnet50', help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for the model')
    parser.add_argument('--bin_thresh', type=float, default=0.3, help='Binarization threshold for postprocessor')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for computation (e.g., "cuda" or "cpu")')

    args = parser.parse_args()
    main(args)
