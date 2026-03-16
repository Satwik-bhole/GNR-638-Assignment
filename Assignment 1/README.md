# GNR-638-Assignment
### 7.1 Prerequisites

* **C++ Compiler** with C++17 support (GCC >= 7 or Clang >= 5)
* **CMake** >= 3.10
* **OpenCV** (C++ development libraries)
* **pybind11** (install via `pip install pybind11` or system package manager)
* **Python 3** with standard libraries

### 7.2 Step-by-Step Commands

```bash
# ==========================================================
# STEP 0: Install Dependencies (Linux/Ubuntu)
# ==========================================================
# Install OpenCV C++ headers and libraries
sudo apt-get update
sudo apt-get install libopencv-dev

# Install pybind11 and CMake via pip
pip install pybind11 cmake

# ==========================================================
# STEP 1: Train/Test Split
# ==========================================================
python train_test_split.py data

# ==========================================================
# STEP 2: Build the C++ Backend with CMake
# ==========================================================
mkdir build
cd build
cmake ..
make
cd ..

# The shared library (my_dl_framework.*.so) will be
# placed in the project root directory.

# ==========================================================
# STEP 3: Train the Model
# ==========================================================
python3 train.py data_1/Train weights/set_1
python3 train.py data_2/Train weights/set_2


# ==========================================================
# STEP 4: Evaluate the Model
# ==========================================================
python eval.py data_1/Test weights/set_1
python eval.py data_2/Test weights/set_2
