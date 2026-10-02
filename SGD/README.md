# Stochastic Gradient Descent for Bike Sharing Demand Prediction

This project explores **Stochastic Gradient Descent (SGD)** for predicting bike rental demand using the Bike Sharing Dataset.

## Objective

The goal is to build a regression model that predicts the logarithm of bike rental count (`cnt_log`) using weather, time, and calendar-related features.

## Dataset

The project uses the **Bike Sharing Dataset**, which contains hourly bike rental information along with:

* Season
* Year
* Month
* Hour
* Holiday
* Weekday
* Working day
* Weather situation
* Temperature
* Feeling temperature
* Humidity
* Windspeed

The target variable is transformed using:

```python
np.log1p(cnt)
```

to reduce the effect of the right-skewed target distribution.

## Data Preprocessing

The notebook includes:

* Removal of irrelevant columns
* Categorical feature analysis
* Continuous feature analysis
* Outlier detection using Z-score and IQR
* One-hot encoding for categorical variables
* Standardization of continuous variables
* Train/test split
* Log transformation of the target variable

## Model

The main model used is:

**Scikit-learn `SGDRegressor`**

Configuration:

* Loss: `squared_error`
* Penalty: `None`
* Learning rate: `constant`
* Initial learning rate (`eta0`): `0.001`
* Maximum iterations: `5000`
* Random state: `42`

## Evaluation

The model is evaluated using:

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

### Current Result

The current configuration produces extremely large errors and a highly negative R² score, indicating that the SGD model is **not converging properly with the current settings**.

This result is kept intentionally as part of the learning process. The next step is to investigate learning-rate selection, feature preprocessing, and SGD convergence.

## Notebook

The complete analysis, preprocessing, model training, and evaluation are available in:

`sgd.ipynb`

## Key Learning

This project demonstrates that using SGD is not simply a matter of fitting the model. **Learning rate, feature scaling, preprocessing, and convergence behavior have a major effect on the final model.**
