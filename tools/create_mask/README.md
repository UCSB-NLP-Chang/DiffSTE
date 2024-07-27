
# Mask Creating Tool

This tool processes images for creating DiffSTE samples using the Doctr library. It detects text regions in an image, randomly crops these regions, and saves the cropped images along with their masks.

## Features

- Allows configuration of various parameters via command-line arguments.
- Well suited for documents.

## Requirements

- Python (tested on 3.10)
- OpenCV
- NumPy
- Matplotlib
- Doctr

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the script with the required parameters:

```bash
python main.py --input_image /path/to/image.jpg --output_dir /path/to/output/dir
```

You can also specify additional parameters as needed:

```bash
python main.py --input_image /path/to/image.jpg --output_dir /path/to/output/dir --arch db_resnet50 --batch_size 4 --bin_thresh 0.5 --device cuda
```

### Command-Line Arguments

- `--input_image`: Path to the input image (required).
- `--output_dir`: Directory to save the output images (required).
- `--arch`: Model architecture (default: `db_resnet50`).
- `--pretrained`: Use pretrained model (default: `True`).
- `--pretrained_backbone`: Use pretrained backbone (default: `True`).
- `--batch_size`: Batch size for the model (default: `2`).
- `--assume_straight_pages`: Assume straight pages (default: `False`).
- `--preserve_aspect_ratio`: Preserve aspect ratio (default: `False`).
- `--symmetric_pad`: Use symmetric padding (default: `True`).
- `--bin_thresh`: Binarization threshold for postprocessor (default: `0.3`).
- `--device`: Device to use for computation, e.g., "cuda" or "cpu" (default: `cuda`).

## Acknowledgements

This tool uses the [Doctr](https://github.com/mindee/doctr) project for document text detection. 

### Citation

```bibtex
@misc{doctr2021,
    title={docTR: Document Text Recognition},
    author={Mindee},
    year={2021},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/mindee/doctr}}
}
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
