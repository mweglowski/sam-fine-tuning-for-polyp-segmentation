![masks](https://i.imgur.com/yOri18g.png)
# SAM Fine-Tuning for Medical Polyp Segmentation

## Overview
This repository contains a Jupyter Notebook implementing a fine-tuning pipeline for Meta's Segment Anything Model (SAM), adapting it for colonoscopy polyp segmentation. The project utilizes a resource-efficient training paradigm by freezing the computationally expensive Vision Transformer (ViT) image encoder and prompt encoder (`requires_grad=False`), restricting gradient updates exclusively to the lightweight mask decoder.

## Technical Architecture & Features
* **Custom Data Pipeline:** Implements a PyTorch `Dataset` and `DataLoader` to handle SAM's strict input tensor formatting, including longest-edge scaling and zero-padding to `[3, 1024, 1024]`.
* **Prompt Engineering:** Supports and evaluates dynamic prompt resolution strategies, specifically Ground Truth (GT) Bounding Boxes (`[B, 4]`) and GT Centroid Points (`[B, N, 2]`).
* **Optimization Strategy:** Utilizes the `AdamW` optimizer (LR: 1e-5, Weight Decay: 1e-4) paired with a composite BCE + Dice Loss function to mitigate severe class imbalance and stabilize Transformer training.
* **Quantitative Benchmarking:** Computes Mean Intersection over Union (mIoU) and Dice Scores to evaluate the fine-tuned decoder against the Zero-Shot SAM Baseline.
* **Failure Mode Analysis:** Automates the extraction and visualization of the top 10 and bottom 10 inference results to formulate hypotheses around edge cases (e.g., specular highlights, low mucosal contrast, tool occlusion).
* **Feature Activation Mapping:** Extracts and interpolates intermediate embeddings (`[1, 256, 64, 64]`) from the frozen ViT to visualize the encoder's innate saliency detection via heatmaps.
* **Interactive Inference UI:** Integrates a local `Gradio` application within the notebook for real-time tensor processing, `torch.no_grad()` inference, and OpenCV-based alpha blending for interactive point/box masking.

## Dataset
The project utilizes the **Kvasir-SEG** dataset. Images and corresponding GT masks are preprocessed dynamically during the `__getitem__` call to meet SAM's expected uniform square input geometry without distorting original aspect ratios.

## Requirements
To execute this pipeline, a Python environment with the following dependencies is required:
* `torch`
* `segment-anything`
* `opencv-python`
* `matplotlib`
* `numpy`
* `pandas`
* `gradio`
* `tqdm`

*Note: A CUDA-enabled GPU is highly recommended for the training loop and real-time interactive inference.*

## Usage
The entire workflow is self-contained within a single Jupyter Notebook.
1. Clone the repository and install the required dependencies.
2. Download the Kvasir-SEG dataset and update the local directory paths within the notebook.
3. Download the baseline SAM checkpoint (`sam_vit_b_01ec64.pth`) from the official Meta repository.
4. Execute the notebook cells sequentially to initialize the `DataLoader`, run the training loop, generate evaluation metrics, and plot the visual analysis.
5. Run the final cell to launch the local Gradio server for interactive testing.

## Results
The fine-tuned mask decoder demonstrates high gains in IoU and Dice metrics over the zero-shot baseline.
