# Stack Overflow Topic Classifier

![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/your_username/stackoverflow-topic-classifier/issues)


![PyTorch Logo](https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png)

A PyTorch-based RNN model for classifying Stack Overflow questions into Spark, Machine Learning, or Security categories.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contribution](#contribution)
- [License](#license)

## Introduction

The goal of this project is to develop an understanding of natural language processing and recurrent neural networks (RNNs) by implementing a topic classification model to categorize Stack Overflow questions into three categories: Spark, Machine Learning, or Security. The project utilizes PyTorch, an open-source machine learning framework, and RNNs, specifically LSTMs or GRUs, for handling sequential text data.

### Pipeline
<img src="assets/pipeline.png"/>

### Dataset and Models

- Dataset: Stack Overflow data containing questions and their associated categories (Spark, Machine Learning, or Security)
- Models: RNN-based model (LSTM and GRU) as the primary approach

## Installation

1. Clone the repository
```bash
git clone https://github.com/your_username/stackoverflow-topic-classifier.git
