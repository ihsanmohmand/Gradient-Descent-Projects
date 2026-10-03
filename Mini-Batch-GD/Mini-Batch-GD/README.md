# Mini-Batch Gradient Descent

This project implements **Mini-Batch Gradient Descent** for regression using Scikit-learn's `SGDRegressor`.

## What I Did

Instead of training the model on the complete dataset at once or using only one sample at a time, I trained the model using **small batches of data**.

A batch of **35 samples** is randomly selected in each iteration and passed to the model using `partial_fit()`.

## Implementation

```python
sgd = SGDRegressor(learning_rate='constant', eta0=0.1)

batch_size = 35

for i in range(100):
    idx = random.sample(range(X_train.shape[0]), batch_size)
    sgd.partial_fit(X_train[idx], y_train[idx])
```

## How It Works

* `SGDRegressor` is used as the learning model.
* `batch_size = 35` means 35 training samples are used for each update.
* `random.sample()` randomly selects the samples for each batch.
* `partial_fit()` updates the model using that batch.
* The process is repeated for **100 iterations**.

## Mini-Batch Gradient Descent

Mini-Batch Gradient Descent is between:

* **Batch Gradient Descent** → uses the entire training dataset for each update.
* **Stochastic Gradient Descent (SGD)** → uses one sample for each update.
* **Mini-Batch Gradient Descent** → uses a small group of samples for each update.

## Tools Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Google Colab / Jupyter Notebook
