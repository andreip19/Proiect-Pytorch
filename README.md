\# 🚀 PyTorch to Native C++ AI Inference Pipeline



\## 📌 Project Overview

A high-performance, end-to-end AI inference pipeline bridging Python and C++. This project trains a Convolutional Neural Network (CNN) in Python, traces it via TorchScript, and executes native, low-latency multi-image inference in C++ using LibTorch.



\## ⚙️ Tech Stack

\* \*\*Languages:\*\* C++20, Python 3.10

\* \*\*Machine Learning:\*\* PyTorch 2.8.0, LibTorch 2.8.0 (C++ API)

\* \*\*Build System \& Tooling:\*\* CMake, MSVC, Visual Studio Code



\## 🧠 Advanced ML Techniques Implemented

\* \*\*Data Augmentation:\*\* Applied `RandomAffine` (rotations, translations, scaling) during PyTorch training to simulate imperfect real-world handwriting and ensure model robustness.

\* \*\*Spatial Invariance \& Optimization:\*\* Added a `MaxPool2d` layer that drastically reduced the model's physical footprint from \~510 KB to \*\*136 KB\*\*, decreasing memory usage and improving inference speed without sacrificing >97% accuracy.

\* \*\*Mathematical Preprocessing:\*\* The Python preprocessing script computes the \*\*Center of Mass\*\* of the custom handwritten digits instead of standard geometric bounding boxes, perfectly aligning custom user input with the mathematical distribution of the original MNIST training dataset.

\* \*\*Sequential Batch Inference:\*\* Native C++ `<filesystem>` implementation allows dynamic loading and processing of entire directories of binary `.bin` tensors seamlessly.



\## 📂 Architecture \& Core Files

1\. `train\_model.py`: Defines the CNN architecture, applies augmentation, trains the model, and exports it as `model\_traced.pt`.

2\. `prepare\_image.py`: Reads raw `.png` images, applies Center of Mass normalization, crops to a 28x28 canvas, and exports flat `uint8` arrays (`.bin`).

3\. `main.cpp`: The native C++ inference engine. Safely loads the TorchScript model via `c10::Error` handling, dynamically scans output directories, normalizes arrays to float tensors `\[1, 1, 28, 28]`, and executes the forward pass.

4\. `CMakeLists.txt`: Configures the C++20 environment, statically targets LibTorch, suppresses third-party STL warnings, and automates DLL copying for the final executable.



\## 🚀 Build and Run (Windows)



\### Prerequisites

\* Python 3.10+ (with PyTorch and Pillow installed)

\* LibTorch 2.8.0 (Release version) extracted to `D:\\libtorch-2.8.0`

\* CMake and MSVC Build Tools



\### Step 1: Preprocess Images

Place any handwritten `.png` digits into the `test\_images/` directory, then run the Python pipeline:

```powershell

python prepare\_image.py

Step 2: Build the C++ Engine
PowerShell
cmake -S . -B build
cmake --build .\build --config Release
Step 3: Run Inference
The CMake configuration automatically copies the required LibTorch DLLs. Simply execute the binary:

PowerShell
.\build\Release\ai_inference.exe

