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
