# DAT255: Deep Learning Engineering

![](assets/DAT255-logo.png)

<!-- <p>
[![kaggle](./assets/kaggle.svg)](https://www.kaggle.com/alexanderlundervold/code) &nbsp;  [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HVL-ML/DAT255/blob/main/) &nbsp; [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/HVL-ML/DAT255/HEAD) &nbsp; [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/HVL-ML/DAT255/tree/main/)
</p> -->

> Material used in the 2025 version of the course DAT255 at HVL. More information can be found on the course website: https://hvl.instructure.com/courses/28951. 

## About the course
:eyes: Deep learning is a sub-field of machine learning and is what both launched and propels the recent surge of interest in artificial intelligence. The course focuses on deep learning and its applications in computer vision, natural language processing, generative AI, and recommendation systems. The methods, tools, and techniques covered by the course are widely applicable, and the course aims to be instructive to anyone wanting to apply deep learning to any task.

:eyes: In addition to a solid understanding of deep learning, the course will provide you with hands-on experience designing and deploying deep learning solutions for practical, real-life problems using state-of-the-art techniques and software frameworks from machine learning, machine learning engineering, and deep learning. You will experience first-hand how deep learning engineering relates to the broader software engineering discipline. Upon completing this course, you'll be well-equipped to tackle challenges in AI-driven software development.

## Content

:point_right: Go to [`/notebooks`](/notebooks) to find the course material. 


## Setup 

The simplest way to run the notebooks is through a cloud service like [https://colab.research.google.com/](Colab). However, if you want to run on your own computer, the setup is relatively simple. The commands below assume a `bash` shell -- if you are on Mac, have a look at these [instructions](https://developer.apple.com/metal/tensorflow-plugin/) (note you need python 3.9 or 3.10), and if you are on Windows, just Google what to do.

1. **Create a virtual environment (optional)**<br>
    If you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) installed, run 
    ```bash
    conda create --name dat255 python=3.12
    conda activate dat255-env
    ```
    otherwise use the built-in virtual environment tool by running
    ```bash
    python3 -m venv dat255-env
    source dat255-env/bin/activate
    ```
2. **Clone our repository**
    ```bash
    git clone git@github.com:HVL-ML/DAT255.git
    cd DAT255
    ```
3. **Install the frameworks we need**
    ```bash
    pip install -r requirements.tex
    ```
4. **Run notebooks**
    ```bash
    cd notebooks
    jupyter notebook
    ```

More in-depth details about installing TensorFlow (the most important and most difficult thing to install) are listed in the [documentation](https://www.tensorflow.org/install/pip).

For help with any of the above, visit the lab on Mondays!

---

> A course from the AI Engineering Group at the Western Norway University of Applied Sciences.
<center>
<a href="https://github.com/HVL-ML">
<img width=40% src="assets/AI-eng.png"></img>
</a>
</center>
