# Stack Overflow Topic Classifier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18ghs9h4NegJUocBamwlfdj8OibSsW8S7)

This project demonstrates the classification of Stack Overflow posts into three categories: "spark", "ml", and "security". The performance of two different recurrent neural network (RNN) architectures, Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU), is compared.

<img src=assets/rnn-pipeline.drawio.png/>

## Table of Contents

- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Diagrams](#diagrams)
- [Usage](#usage)

## Dataset

The dataset used in this project is located in the `datasets/SO.csv` file. It contains Stack Overflow post titles and their corresponding labels ("spark", "ml", or "security").

## Model Design

The following RNN architectures are implemented and compared:
1. **LSTM Classifier**
2. **GRU Classifier**

## Results

After training and evaluating both models, a comprehensive comparison of their performance is provided using the following metrics:

- Receiver Operating Characteristic (ROC) curves
- Area Under the Curve (AUC) values
- Confusion matrices
- Classification reports

Hyperparameter tuning is performed to improve the performance of the selected model, and the updated model is re-evaluated using the same metrics.

## Usage

The entire project is implemented in a Jupyter Notebook. To run the project, follow these steps:

1. Clone the repository.
2. Open the Jupyter Notebook.
3. Run the notebook cells in order, starting from the top.

Ensure the required dependencies are installed.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---


