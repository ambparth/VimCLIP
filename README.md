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

In the experiments, we make use of the combination of Conceptual Datasets (CC) of the 3M and 12M variants in the case of the Cross Modal Retrieval Tasks (image-to-text, text-to-image). Similarly in the case of the Zero-shot classification, the different variants of the ImageNet dataset are used; validation set, rendition set, adversarial set and the sketch set. 

The dataset for the CC3M variant can be found in [here](https://ai.google.com/research/ConceptualCaptions/download). Further, in order to get to a corresponding `.csv` format, 

`python src/data/gather_cc.py [path/to/cc3m/images/] [path/to/cc3m_train.tsv] [path/to/cc3m_val.tsv]`

Similarly for the CC12 variant, the following [link](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc12m.md) can be used. In otder to get the similar corresponding `.csv` format, 

`python src/data/gather_cc12m.py [path/to/cc12m/images/] [path/to/cc12m.tsv]`

as the downloadable data are in the form of URL, the data availability purely resides on the availability of the URL. At the time of our download, the combined dataset of CC12M+CC3M were around 9.1M. 

Similarly in the case of the zero-shot classification, the Imagenet dataset variants are used. 
* ImageNet validation dataset (imagenet-val) - [link](https://www.image-net.org/download.php)
* ImageNet Rendition dataset (imagenet-r) - [link](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar)
* ImageNet Adversarial dataset (imagenet-a) - [link](https://www.kaggle.com/datasets/paultimothymooney/natural-adversarial-examples-imageneta)
* Imagenet Sketch dataset (imagenet-sketch) - [link](https://www.kaggle.com/datasets/wanghaohan/imagenetsketch)

## Training 
In these experiments, three variants of the models are used for training. The base (B), small (S) and the tiny (T) variants. For running the training on the above mentioned dataset, 

`bash script/baseline/ViT_<v>_16.sh ` for training the vision transformer based models (ViT as the vision encoder) 

`bash script/baseline/VimCLIP_<v>.sh` for training the vision mamba based VimCLIP model. 

where `<v>` is the variant of the model (either T, S or B) 

## Evaluation 
The evaluation is done on the validation set of the CC3M dataset for the cross modal retrieval tasks and on the ImageNet data variants in the case of the zero shot classfication. These are placed in the `evaluations` folder.

`bash script/evaluation/ViT_<v>_16.sh ` for training the vision transformer based models (ViT as the vision encoder) 

`bash script/evaluation/VimCLIP_<v>.sh` for training the vision mamba based VimCLIP model. 

## Acknowledgement
Our codes are built over [CLIP-KD](https://github.com/winycg/CLIP-KD) and [Vim](https://github.com/hustvl/Vim). If you find this repository useful for your work and application, please beep out from here, as i have no citation at the moment. 

