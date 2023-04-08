# Stack Overflow Topic Classifier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18ghs9h4NegJUocBamwlfdj8OibSsW8S7)

PyTorch-based RNN model for classifying Stack Overflow questions into Spark, Machine Learning, or Security categories.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contribution](#contribution)
- [License](#license)

## Introduction

The goal of this project is to develop an understanding of natural language processing and recurrent neural networks (RNNs) by implementing a topic classification model to categorize Stack Overflow questions into three categories: Spark, Machine Learning, or Security using PyTorch.

### Pipeline
```mermaid
journey
    title Machine Learning Pipeline
    section Data Collection
        User navigates to data source: 5: User
        User downloads data: 2: User
        Data is saved to local directory: 1: System
    section Data Preprocessing
        Data is cleaned and filtered: 3: System
        Text is tokenized: 2: System
        Text is vectorized: 4: System
    section Model Training
        Model is designed and compiled: 3: System
        Hyperparameters are tuned: 5: System
        Model is trained on data: 4: System
    section Model Evaluation
        Model is evaluated on test data: 4: System
        Results are analyzed: 2: User
    section Model Deployment
        Model is deployed to production: 5: System
        Users can access model predictions: 3: User
```

### Dataset and Models

- Dataset: Dataset: Stack Overflow data containing questions and their associated categories (Spark, Machine Learning, or Security). The dataset can be found on Kaggle at https://www.kaggle.com/datasets/jashdubal/stack-overflow-classification.
- Models: RNN-based model (LSTM and GRU) as the primary approach.

## Installation

_Coming soon._

## Usage

_Coming soon._

## Contribution

Contributions are welcome! Please feel free to submit a Pull Request or create an Issue to discuss new features or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

