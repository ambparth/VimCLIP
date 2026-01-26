# VimCLIP

This repository here contains the relevant codes of VimCLIP [VimCLIP: A Vision Mamba Based Multimodal Approach for Retrieval and Zero-Shot Classification Tasks]. 

## Installation Procedures
The codes are run with a python environement version 3.10, and with CUDA version 12.4
* The Python Environment version 3.10.18
  * `conda create --name mamba python=3.10.18`
* Installation of `torch` version `v2.1.1+cu12.4` and `torchvision` libraries:
  * `pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu124`

The successful installation of Torch can be verified if `torch.cuda.is_available()` is returned as `True`  

* Then install all the other necessary packages using the command, `pip install -r requirements.txt`


* Further scroll to the `Vim` directory using the command, `cd VimCLIP/src/Vim/`
  *  `python3 causal-conv1d/setup.py --install`
  *  `python3 mamba-1p1p1/setup.py --install`

## Dataset Preparation
