# Stack Overflow Topic Classifier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open in Jupyter Notebook](https://img.shields.io/badge/Open%20in-Jupyter%20Notebook-orange)](https://github.com/jashdubal/stackoverflow-classification/blob/main/SO_notebook.ipynb)

This project demonstrates the classification of Stack Overflow posts into three categories: "spark", "ml", and "security". The performance of two different recurrent neural network (RNN) architectures, Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU), is compared.

<img src=assets/rnn-pipeline.drawio.png/>

## Table of Contents

- [Background](#background) 
- [Dataset](#dataset)
- [How to Run](#how-to-run)
- [Model Design](#model-design)
- [Training](#training)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [License](#license)

## Background

This repository contains the following code files:
- [`SO_notebook.ipynb`](SO_notebook.ipynb): Jupyter Notebook that contains the code for training and evaluating a machine learning model on the Stack Overflow dataset.
- [`dataset/SO.csv`](dataset/SO.csv): Stack Overflow dataset used to train and evaluate the machine learning model in SO-notebook.ipynb.

## Dataset

The dataset used in this project is located in the [`dataset/SO.csv`](dataset/SO.csv) file. It contains Stack Overflow post titles and their corresponding labels ("spark", "ml", or "security").

The dataset consists of 150,000 entries with no missing values, and includes two columns: 'Title' and 'Label'. The data types for both columns are objects (strings).

The target distribution of the dataset is balanced, with each label having 50,000 samples:
- spark: 50,000
- ml: 50,000
- security: 50,000

## How to Run

The entire project is implemented in a Jupyter Notebook. To run the project, follow these steps:

1. Clone the repository.
2. Install the required dependencies using pip. You can do this by running the following command:

```shell

pip install torch numpy pandas scikit-learn seaborn matplotlib nltk
```

3. Open the Jupyter Notebook `SO-notebook.ipynb` in Jupyter Notebook or JupyterLab.
4. Follow the instructions provided in the notebook to train and evaluate the LSTM and GRU models on the Stack Overflow dataset.

Note: Running the entire notebook may take up to 3 hours, depending on your machine's hardware specifications.

## Model Design

Two RNN architectures are implemented and compared:

1. **LSTM Classifier**: An LSTM-based RNN model to classify Stack Overflow post titles.
2. **GRU Classifier**: A GRU-based RNN model to classify Stack Overflow post titles.

Both models are defined using the PyTorch framework, with custom classes `LSTMClassifier` and `GRUClassifier`.

## Training

The training process is implemented using a custom `train_and_evaluate()` function. The training loop consists of the following steps:

1. Set the model to training mode.
2. Iterate over the training data in mini-batches.
3. Perform forward pass.
4. Calculate the loss using CrossEntropyLoss.
5. Perform backpropagation to compute gradients.
6. Update model parameters using Adam optimizer.

## Hyperparameter Tuning

The hyperparameters of interest in this project are the hidden dimension and dropout rate. By experimenting with different values for these hyperparameters, we can improve model performance.

## Results

In selecting RNN models, LSTM and GRU were considered beacuse they are both popular types of RNNs that excel at text classification tasks. I decided to compare the performance between the two models through a series comparison of ROC curves, confusion matrices, and classification reports.

The slightly higher average AUC of **0.9359** in the LSTM ROC curve tells us that this model slightly outperforms GRU model when it comes to comparison between all three classes.

Confusion matrices and classification report also slightly favour LSTM over GRU.

### Receiver Operating Characteristic (ROC) curves

| LSTM Model | GRU Model | Tuned LSTM Model (ndim=256, dr=0.3) |
|------------|-----------|-----------------------------------------------|
| ![LSTM ROC](assets/lstm_roc.png) | ![GRU ROC](assets/gru_roc.png) | ![Tuned LSTM ROC](assets/tuned_lstm_roc.png) |

### Confusion matrices

| LSTM Model | GRU Model | Tuned LSTM Model (ndim=256, dr=0.3) |
|------------|-----------|-----------------------------------------------|
| ![LSTM CM](assets/lstm_cm.png) | ![GRU CM](assets/gru_cm.png) | ![Tuned LSTM CM](assets/tuned_lstm_cm.png) |


### Classification report

```
LSTM Model Performance:
              precision    recall  f1-score   support

       spark     0.9111    0.8986    0.9048     10000
          ml     0.9085    0.9051    0.9068     10000
    security     0.9239    0.9400    0.9319     10000

    accuracy                         0.9146     30000
   macro avg     0.9145    0.9146    0.9145     30000
weighted avg     0.9145    0.9146    0.9145     30000
```

```
GRU Model Performance:
              precision    recall  f1-score   support

       spark     0.8998    0.9075    0.9036     10000
          ml     0.9018    0.9014    0.9016     10000
    security     0.9392    0.9315    0.9353     10000

    accuracy                         0.9135     30000
   macro avg     0.9136    0.9135    0.9135     30000
weighted avg     0.9136    0.9135    0.9135     30000

```

```
LSTM Tuned Model Performance:
 precision    recall  f1-score   support

       spark     0.8868    0.9228    0.9044     10000
          ml     0.9127    0.9021    0.9074     10000
    security     0.9530    0.9254    0.9390     10000

    accuracy                         0.9168     30000
   macro avg     0.9175    0.9168    0.9169     30000
weighted avg     0.9175    0.9168    0.9169     30000
```

## License

This project is licensed under the [MIT License](LICENSE).

---

