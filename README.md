#### COMP 551 Final Project - McGill Univeristy - Professor W. Hamilton   
**Baseline Reproduction for Fasion MNIST** May 2019  
Team 16: [@Blair D.](https://github.com/BlairKH)
[@Suki L.](https://github.com/SiqiLiu43)
[@Ellen C.](https://github.com/3llen)

# Abstraction
This project attempts to reproduce as well as improve the baseline accuracy from Bhatnagar,
Ghosal and Kolekar’s paper [[Link on IEEE]](https://ieeexplore.ieee.org/abstract/document/8313740) for the classification task of fashion article images obtained from the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).   
     
In the first part of the project, we attempt to reproduce the SVC baseline for the classification task, and obtained an accuracy score of **0.8986**, which is slightly higher than the accuracy of **0.8970** in the original paper. Then, we tried to improve the SVC baseline by hyper-parameter tuning, involving different SVC kernel trials, different C-values and different gamma values of the SVC models. Our best performing model with polynomial kernel (degree = 2 and C = 10.0) achieved a test accuracy of **0.8950** whereas the best performing classifier with the RBF kernel (gamma = 0.001 and C = 10.0) resulted in an accuracy of **0.8970**. We also implemented a Convolutional Neural Network model that has around 1% increase in accuracy compared to the CNN models in the original paper, with a test accuracy score of **0.9355**.


# Introduction
Clothing has long been considered as a reflection on cultural identity, lifestyle, gender and social status [1]. Fashion trend is also an important descriptor that reveals a society's appreciation of beauty as well as its underlying cultural values [2]. Thus, many possible applications arise when fashion meets machine learning. For example, the discovery of similar fashion items can be facilitated by predicting the class that item belongs to [3]. Moreover, real-time clothing recognition can be convenient in the surveillance context [4] which, furthermore, can be advantageous in searching for missing population.

Image classification involves associating an input image with a specified image class; it is considered as an essential problem in computer vision[5].  This project focuses on the classification of fashion article images (Fashion-MNIST). Unlike the original MNIST dataset that offers only 10 possibilities (i.e. 10 digits), the Fashion-MNIST provides a much more diverse classification problem [6].

In this project, we first reproduced the SVC baseline for the Fashion-MNIST classification problem in Bhatnagar, Ghosal and Kolekar's paper [7]. The SVC baseline in the original paper achieved an accuracy score of **0.8970**, whereas our attempt achieved a slightly higher accuracy score of **0.8986**. Then, we used hyper-parameter tuning to attempt to improve our model. Specifically, we conducted several trials using different SVC kernels, different C-values of the SVC models, as well as different gamma-values.
Our second task in this project consists of suggesting a new improved baseline for this image classification problem. We chose to implement a simple CNN model that maximizes accuracy  Inspired by Danial Khosravi's approach to the problem, we also implemented a simple CNN model that achieved an accuracy of  **0.9355** [8].
