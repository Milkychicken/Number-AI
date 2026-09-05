"""
A file to test PyTorch and Cuda availability/
install success when setting up new systems

Setup for new devices can be found in the README file

Run this file to test the installation
"""

import torch as t

print(t.__version__)
print("cuda available!" if t.cuda.is_available() else "cuda not available (missing graphics card?)")
if t.cuda.is_available():
    print(t.cuda.get_device_name(0))