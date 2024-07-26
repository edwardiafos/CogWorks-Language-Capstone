# CogWorks-Language-Capstone
## Brought to you by We Love Bytes
![image](https://github.com/user-attachments/assets/7fae1fdf-7ea8-4563-8c20-559208ea15e1)
<br/>
<br/>

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [Contributors](#contributors)

## Introduction
For the week 3 Capstone Project for BWSI, we made a text-to-image generator where given a prompt/caption, the user would be returned an image of what they had inputted. 
Our model makes use of the [COCO (Common Objects in Context) dataset](https://cocodataset.org/#home) to receive both captions and images to train our Natural Language Processing Model. 
<br/>
*Note: This project was created as part of BWSI CogWorks Week2 Vision Capstone project.*
*Presented by Team We Love Bytes 2024.*
<br/>
<br/>
## Features
These are some important features that our program can accomplish:
- [x] A functioning User Interface that allows for Light/Dark mode
- [x] A database of over 80k images each with captions
- [x] A trained NLP model able to effectively match captions to images
- [x] A database containing image urls and semantic embeddings
- [x] An option to automatically remove stopwords (i.e. words that if taken away, generally keeps meaning of sentence) from your input

## Installation

Before the installation of this program, we recommend you utilize a conda environment.
For more information about conda installation, visit https://conda.io/projects/conda/en/latest/user-guide/install/index.html.

To set up the optimal conda environment, run the appropriate command and replace `env_name` with whatever you like.

For both MacOS and Windows
```
conda create -n week3 python=3.8 jupyter notebook numpy matplotlib numba scikit-learn nltk=3.6.5
```
```
conda install -n week3 -c conda-forge python-graphviz
```

We will need PyTorch as well. If on Windows or Linux, run
```
conda install -n week3 pytorch torchvision cpuonly -c pytorch
```

If on MacOS, run
```
conda install -n week3 pytorch torchvision -c pytorch
```

Make sure to activate this conda environment by running
```
conda activate week3
```
Once the new environment is activated, install MyGrad, MyNN, Noggin, and Gensim, by running:
```
pip install mygrad mynn noggin gensim cogworks-data
```
Navigate back to your original parent folder and clone this repository.
```
git clone https://github.com/edwardiafos/CogWorks-Language-Capstone.git
```
Navigate to our folder.
```
cd CogWorks-Vision-Module-Capstone
```
Install all dependencies.
```
pip install -r requirements.txt
```
And run main.py to start the program.
```
python main.py
```
## Usage
![image](https://github.com/user-attachments/assets/172b54cd-6cd6-4fe3-b6aa-ddf12db7116b)
![image](https://github.com/user-attachments/assets/aab13d93-014d-4045-aeb8-eb0cd8d70cad)




> Note: It is important to use this technology with ethics and consideration in mind. Privacy is a right.

## Configuration
We already optimized the configuration of this program. 

## Contributors
- Zoe Granadoz
- Edwardia Fosah
- Bryan Wang
- Ye Yint Hmine
- Manya Tandon


