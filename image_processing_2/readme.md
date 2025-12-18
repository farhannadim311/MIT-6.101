# Content-Aware Image Resizing Engine (Seam Carving)

A **Python-based implementation of Seam Carving** for content-aware image resizing. This project implements the core algorithm **from scratch**, without relying on high-level computer vision libraries (e.g., OpenCV), demonstrating a fundamental understanding of **pixel-level manipulation, energy mapping, and dynamic programming**.

---

## Overview

Unlike standard scaling techniques (cropping or uniform resizing), **Seam Carving** adjusts an image’s aspect ratio by identifying and removing paths of *least visual importance*—called **seams**. This preserves important structures and subjects while reducing image dimensions.

This technique is foundational for **responsive media layouts**, enabling images to adapt to different displays (mobile, desktop, spatial computing canvases) without losing semantic content.

---

## Key Features

- **Energy Mapping**
  - Sobel edge detection (gradient magnitude) to estimate pixel importance
- **Dynamic Programming**
  - Efficient computation of the **Cumulative Energy Map (CEM)**
- **Backtracking Algorithm**
  - Identifies minimum-energy seams from bottom to top
- **Filter Pipeline**
  - Custom implementations of:
    - Gaussian blur
    - Unsharp masking (sharpening)
    - Color channel correlation

---

## Technical Implementation

- **Language:** Python 3  
- **Dependencies:** Minimal  
  - `PIL / Pillow` used *only* for image I/O  
  - All image processing logic written in **pure Python**
- **Performance**
  - Optimized 1D array representation for 2D image data  
  - Reduced memory overhead during seam removal
## Usage

### 1. Run the script
This script can be imported as a module or run directly to process images.

```python
from lab import load_color_image, save_color_image, seam_carving

# Load original image
image = load_color_image('input.png')

# Remove 100 columns (pixels) content-awarely
resized = seam_carving(image, ncols=100)

# Save result
save_color_image(resized, 'output.png')
