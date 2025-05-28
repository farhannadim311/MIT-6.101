# MIT-6.101
This repository holds all the Problem Sets I solved as I took MIT's 6.101: Fundamentals of Programming (Spring 2025).

# 📁 PSet 1: Audio Signal Processing (6.101 Lab)

This problem set explores foundational audio signal processing techniques using pure Python. The assignment emphasizes core concepts like echo, convolution, mixing, and stereo channel manipulation — all implemented without external libraries to build a low-level understanding of digital audio.

---

## 🎯 Objectives

- Manipulate mono and stereo `.wav` files at the sample level
- Implement convolution-based filters from scratch
- Simulate acoustic effects like **echo**, **bass boost**, and **panning**
- Practice combining and subtracting audio tracks to achieve effects like **vocal removal**
- Reinforce Python fundamentals with real-world audio applications

---

## 🔧 Setup

- **Language**: Python 3
- **Modules used**: `wave`, `struct`
- **Input audio**: 16-bit `.wav` files located in the `/sounds/` directory

> No external libraries (like NumPy or SciPy) are used.

---

## 🛠 Implemented Functions

### `backwards(sound)`
Reverses the order of samples in a mono sound dictionary.

### `mix(sound1, sound2, p)`
Mixes two mono sounds based on a linear blend factor `p`.

### `echo(sound, num_echoes, delay, scale)`
Adds delayed, scaled copies of the input sound to simulate an echo effect.

### `convolve(sound, kernel)`
Applies a convolution-based transformation using a user-defined kernel.

### `bass_boost_kernel(n_val, scale)`
Constructs a low-pass filter kernel and enhances bass via scaled self-addition.

### `remove_vocals(sound)`
Removes vocals by subtracting the right channel from the left in a stereo file.

### `pan(sound)`
Applies stereo panning from left to right across the duration of the track.

---

## 📂 Directory Structure

```
.
├── lab.py                 # Core implementation file
├── sounds/                # Input WAV files (e.g., lookout_mountain.wav)
├── r.wav                  # Output from remove_vocals()
├── bass.wav               # Output from bass_boost_kernel()
├── echo.wav               # Output from echo()
```

---

## 🚀 Example Usage

```python
from lab import load_wav, echo, write_wav

s = load_wav("sounds/chickadee.wav")
result = echo(s, num_echoes=5, delay=0.6, scale=0.3)
write_wav(result, "echoed.wav")
```

---

## 🧠 Learning Takeaways

- Understand and implement core DSP concepts (echo, filtering, delay)
- Handle WAV audio data without helper libraries
- Gain practical experience with sample-level stereo audio processing
- Write modular, testable Python code for real-world media tasks

---

## ✅ Evaluation Notes

- All functions return **new** sound dictionaries; inputs remain unchanged
- Handles both **mono** and **stereo** formats appropriately
- Output WAV files are properly normalized and clamped to prevent clipping

---

> This problem set provides a hands-on foundation in digital audio programming — ideal preparation for more advanced effects, real-time audio processing, and media tools.

## 📁 PSet 2: Image Processing (6.101 Lab)

A deep-dive into classic digital-image pipelines implemented **from scratch** — no NumPy, no SciPy — just Python, lists, and the ☕ you’ll need to debug correlation math at 2 a.m.  

### 🎯 Objectives
* **Understand digital images** as width × height pixel dictionaries (`{"pixels": …}`)  
* Manipulate greyscale images at the per-pixel level (brightness 0–255)  
* Implement **higher-order functional filters** (`apply_per_pixel`)  
* Build a **general correlation engine** with configurable edge policies (`zero | extend | wrap`)  
* Create real-world filters:  
  * Inversion  
  * Box-blur (arbitrary kernel sizes)  
  * Unsharp-mask sharpening  
  * Sobel **edge detection**  
* Reinforce Python design patterns: helper functions, pure functions, unit tests with `pytest`

---

## 🛠 Implemented Functions

### `apply_per_pixel(image, func)`
Applies a function to every pixel in the image, returning a new image.

### `inverted(image)`
Inverts pixel brightness values (i.e., `pixel = 255 - pixel`).

### `correlate(image, kernel, boundary)`
Applies a kernel to the image using correlation and one of three edge behaviors: `"zero"`, `"extend"`, or `"wrap"`.

### `round_and_clip_image(image)`
Rounds and clamps all pixel values between 0 and 255.

### `blurred(image, n)`
Applies a box blur using an `n x n` averaging kernel.

### `sharpened(image, n)`
Applies an unsharp mask using a blurred version of the image.

### `edges(image)`
Detects edges using the Sobel operator (combining horizontal and vertical gradients).

## 🧠 Learning Takeaways

- Understood and implemented correlation/convolution from scratch
- Visualized and compared edge-handling modes: `zero`, `extend`, and `wrap`
- Built modular, reusable Python functions for pixel-based image processing
- Practiced test-driven development using `pytest`
- Applied mathematical concepts like gradient magnitude, averaging, and unsharp masking
- Reinforced Python fundamentals by manipulating nested lists and dictionaries directly
- Gained practical insight into how Photoshop/GIMP-like filters work under the hood
---


# 🧪 PSET3: Image Processing Lab 2

This project is a full implementation of the MIT 6.101 Lab: **Image Processing Part 2**, extending a basic image manipulation toolkit with advanced operations including convolutional filters, edge detection, sharpening, seam carving, and a custom creative filter.

---

## ✅ Features Implemented

### 📦 Core Image Utilities
- `oned_to_twod` / `flatten`: Converts between 1D and 2D image pixel representations.
- `get_pixel` / `set_pixel`: Boundary-safe access and update for image pixels.
- `apply_per_pixel`: Applies functions to each pixel in a greyscale image.

---

### 🎨 Filters and Effects

- **Inversion**: Simple pixel-wise color inversion.
- **Blurring**: Box blur using correlation with uniform kernels of variable size.
- **Sharpening**: Unsharp masking with custom kernel sizes.
- **Edge Detection**: Sobel filter to extract vertical and horizontal gradients.

---

### 🌈 Color Image Processing
- Grayscale extraction by RGB channel.
- `combine_colors`: Recombines individual channel images into a color image.
- Color wrappers for greyscale filters.

---

### 🧰 Higher-Order Filters
- `make_blur_filter`, `make_sharpen_filter`: Generate parameterized filter functions.
- `filter_cascade`: Combine multiple filters into one seamless transformation pipeline.

---

### ✂️ Seam Carving
- Dynamic programming to compute **minimum energy seams**.
- Functions implemented:
  - `compute_energy`
  - `cumulative_energy_map`
  - `minimum_energy_seam`
  - `image_without_seam`
  - `seam_carving` (removes multiple vertical seams)

---

### 💡 Custom Filter: Ripple Distortion
A radial distortion effect that simulates water ripple waves emanating from the center of the image.

```python
def ripple(image, frequency=5, amplitude=5):
    """
    Applies a ripple distortion to a color image.
    Each pixel is displaced radially using a sine-based function.
    """
```

---

## 📁 Directory Structure

```
image_processing_lab/
├── lab.py                # Main lab implementation
├── test_images/
│   └── twocats.png       # Sample image used for testing
├── twocats_carved.png    # Output of seam carving
├── twocats_rippled.png   # Output of ripple filter
└── README.md             # This documentation file
```

---

## 🚀 How to Run

1. Install dependencies:
```bash
pip install pillow
```

2. Run the main script:
```bash
python3 lab.py
```

Make sure `test_images/twocats.png` exists or change the path in the script.

---

## 📌 Notes

- Seam carving modifies image width while preserving content structure by removing least important pixels.
- All filters are implemented from scratch using only allowed libraries (`math`, `os`, and `PIL`).
- Boundary handling strategies include **zero**, **extend**, and **wrap** for convolution-based operations.

---

## 🧠 Learning Outcomes

- Deep understanding of pixel-based image manipulation
- Working with filters, kernels, and convolution from first principles
- Experience with dynamic programming (seam carving)
- Creativity in developing new visual effects

---
