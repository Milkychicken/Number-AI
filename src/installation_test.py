"""
A file to test PyTorch and Cuda availability/
install success when setting up new systems

--- Setup for new devices ---
1. create a python virtual environment
(in the project folder:) python -m venv venv
2. install pytorch via pip
./.venv/bin/pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
(link may need to be adjusted for different versions)
3. Run this file to test the installation
"""

import torch as t

print(t.__version__)
print("cuda available!" if t.cuda.is_available() else "cuda not available (missing graphics card?)")