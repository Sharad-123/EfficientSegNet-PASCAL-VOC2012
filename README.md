# EfficientSegNet: Lightweight Semantic Segmentation

A lightweight encoder-decoder CNN for 21-class semantic segmentation on the PASCAL VOC 2012 dataset. The model uses depthwise-separable convolutions to reduce computational cost.

## Project Files

- `group_11.ipynb` — complete implementation with saved outputs
- `model.py` — EfficientSegNet architecture
- `inference.py` — inference script
- `best_model.pth` — trained model weights
- `requirements.txt` — Python dependencies

## Model Details

- Dataset: PASCAL VOC 2012
- Number of classes: 21, including background
- Parameters: 14,512
- Computational cost: approximately 0.1813 GFLOPs
- Loss functions: Cross-Entropy and Dice loss
- Input resolution: 300 × 300

## Recorded Results

- Internal validation Dice score: approximately 0.9245
- Separate 100-image evaluation Dice score: approximately 0.4961
- Final training loss: approximately 2.48

These are experimental results saved in the course notebook and are not official PASCAL VOC test-server results.

## Installation

```bash
pip install -r requirements.txt
```

## Notebook

Open `group_11.ipynb` to view the complete implementation and saved outputs. Run the cells sequentially only if you want to reproduce the experiment.

## Inference

```bash
python inference.py --in_dir path/to/images --out_dir path/to/output
```

The script loads `best_model.pth` and saves predicted masks in the output directory.

## Academic Context

Developed as a Group 11 course project on lightweight semantic segmentation.
