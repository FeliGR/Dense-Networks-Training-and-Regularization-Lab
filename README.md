# Image Classification with MLPs

This repository contains materials for a lab exercise on image classification using Multi-Layer Perceptrons (MLPs) with the MNIST dataset. The goal of this lab is to build a neural network that can classify handwritten digits from images. 

## Repository Structure

- `STUDENT_Lab_MLP.ipynb`: The main Jupyter notebook containing the exercises and code for building and training MLPs on the MNIST dataset.
- Additional files: More information on the contents of the other files will be added soon.

## Lab Description

### Objective

This lab guides students through implementing an image classifier using MLPs. Starting with a simple multi-class logistic regressor, we move towards a more complex MLP architecture, exploring key machine learning concepts such as softmax, activation functions, optimization, and model evaluation.

### Dataset

We use the MNIST dataset, which consists of 28x28 pixel greyscale images of handwritten digits (0-9).

### Key Components

- **Data Loading**: Use of `torchvision` to load and preprocess the MNIST dataset.
- **Model Architecture**: Building a multi-class logistic regression model and extending it to a 3-layer MLP with ReLU activations.
- **Training and Evaluation**: Training the model with stochastic gradient descent, computing loss, and evaluating model performance on the test set.
- **Visualization**: Visualizing the weight matrices, activations, and gradients to gain insights into the model's behavior.

## Requirements

- `Python 3.x`
- `PyTorch`
- `torchvision`
- `matplotlib`
- `numpy`

Install the required dependencies using:

```bash
pip install torch torchvision matplotlib numpy
```

## How to Run

1. Clone this repository.
2. Open `STUDENT_Lab_MLP.ipynb` in Jupyter Notebook or Jupyter Lab.
3. Follow the instructions in the notebook and run the cells to complete the lab exercises.

## Acknowledgments

This lab was prepared by **Leire Paz** (lpaz@pa.uc3m.es) and **Alejandro Lancho** (alancho@ing.uc3m.es), based on original material by **Pablo M. Olmos**. Some of the content is inspired by [Facebook's Deep Learning Course on Udacity](https://www.udacity.com/course/deep-learning-pytorch--ud188).
