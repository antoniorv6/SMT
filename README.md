<p align='center'>
  <a href='https://praig.ua.es/'><img src='https://i.imgur.com/Iu7CvC1.png' alt='PRAIG-logo' width='100'></a>
  <a href='https://www.litislab.fr/'><img src='graphics/Litis_Logo.png' alt='LITIS-logo' width='100'></a>
</p>

<h1 align='center'>Sheet Music Transformer: End-To-End Optical Music Recognition Beyond Monophonic Transcription</h1>

<h4 align='center'><a href='https://arxiv.org/abs/2402.07596' target='_blank'>Full-text preprint</a>.</h4>

<p align='center'>
  <img src='https://img.shields.io/badge/python-3.9.0-orange' alt='Python'>
  <img src='https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white' alt='PyTorch'>
  <img src='https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white' alt='Lightning'>
  <img src='https://img.shields.io/static/v1?label=License&message=MIT&color=blue' alt='License'>
</p>

<p align="center">
  <a href="https://huggingface.co/collections/antoniorv6/sheet-music-transformer-66b9c7cd447411b9c0acdce0"><img align="center" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg"></a>
</p>

<p align='center'>
  <a href='#about'>About</a> •
  <a href='#how-to-use'>How To Use</a> •
  <a href='#citations'>Citations</a> •
  <a href='#acknowledgments'>Acknowledgments</a> •
  <a href='#license'>License</a>
</p>

## Updates
- The model has been updated with the HuggingFace Transformers 🤗 library!
- Usage instructions included!
- The paper was accepted at **ICDAR 2024**!

## About

This GitHub repository contains the implementation of the Sheet Music Transfomrmer (SMT), a novel model for Optical Music Recognition (OMR) beyond monophonic level transcription. Unlike traditional approaches that primarily leverage monophonic transcription techniques for complex score layouts, the SMT model overcomes these limitations by offering a robust image-to-sequence solution for transcribing polyphonic musical scores directly from images.

<p align="center">
  <img src="graphics/SMT.jpg" alt="content" style="border: 1px solid black; width: 800px;">
</p>

# Project setup
This implementation has been developed in Python 3.9, PyTorch 2.0 and CUDA 12.0. 

It should work in earlier versions.

To setup a project, run the following configuration instructions:

### Python virtual environment

Create a virtual environment using either virtualenv or conda and run the following:

```sh
git clone https://github.com/antoniorv6/SMT.git
pip install -r requirements.txt
mkdir Data
```

### Docker
If you are using Docker to run experiments, create an image with the provided Dockerfile:

```sh
docker build -t <your_tag> .
docker run -itd --rm --gpus all --shm-size=8gb -v <repository_path>:/workspace/ <image_tag>
docker exec -it <docker_container_id> /bin/bash
```
# Hey, I just want to use the SMT!
Using the SMT for transcribing scores is very easy, thanks to the HuggingFace Transformers 🤗 library. Just implement the following code and you will have the SMT up and running for transcribing excerpts!
```python
import torch
import cv2
from data_augmentation.data_augmentation import convert_img_to_tensor
from smt_model import SMTModelForCausalLM

image = cv2.imread("sample.jpg")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SMTModelForCausalLM.from_pretrained("antoniorv6/<smt-weights>").to(device)

predictions, _ = model.predict(convert_img_to_tensor(image).unsqueeze(0).to(device), 
                               convert_to_str=True)

print("".join(predictions).replace('<b>', '\n').replace('<s>', ' ').replace('<t>', '\t'))

```

# Data

The datasets created to run the experiments are [publicly available](https://grfia.dlsi.ua.es/sheet-music-transformer/) for replication purposes.

Once the ```.tgz``` files are downloaded, store them into the ```Data``` folder, getting the following folder structure:

```
  ├── data
        │   ├── GrandStaff
        │   │   ├──grandstaff_dataset
                ├──partitions_grandstaff
            ├── Quartets
            │   ├──quartets_dataset
                ├──partitions_quartets
```

# Train
These experiments run under the Weights & Biases API and the ```JSON``` config. To replicate an experiment, run the following code:

```sh
wandb login
python train.py --config <config-path>
```
The config files are located in the ```config/``` folder, depending on the executed config file, a specific experiment will be run.

You can make your own config files to train the SMT on your own data! Just, please, if you are using this code, **follow the specified data specification and configuration files structure that is specified in this document**

## Citations

```bibtex
@misc{riosvila2024SMT,
	title        = {Sheet Music Transformer: End-To-End Optical Music Recognition Beyond Monophonic Transcription},
	author       = {Antonio Ríos-Vila and Jorge Calvo-Zaragoza and Thierry Paquet},
	year         = 2024,
	eprint       = {2402.07596},
	archiveprefix = {arXiv},
	primaryclass = {cs.CV}
}
```

## Acknowledgments

This work is part of the I+D+i PID2020-118447RA-I00 ([MultiScore](https://sites.google.com/view/multiscore-project)) project, funded by MCIN/AEI/10.13039/501100011033. Computational resources were provided by the Valencian Government and FEDER funding through IDIFEDER/2020/003.

## License

This work is under a [MIT](LICENSE) license.
