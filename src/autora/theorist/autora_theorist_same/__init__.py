from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from expression_checker import ExpressionChecker

class SAMERegressor(BaseEstimator):
    """
    Theorist to fit the different conditional data efficiently using Linear Regression, Partial Brute Forcing using Pearson coefficients for correlation
    """

    def __init__(self):
      self.linear_with_constant = LinearRegression(fit_intercept=True)
      self.linear_without_constant = LinearRegression(fit_intercept=False)

    def calculate_aic(self, model, x, y):
      """Calculate the Akaike Information Criterion (AIC)."""
      # Predict using the model
      y_pred = model.predict(x)
      # Compute the residual sum of squares (RSS)
      rss = np.sum((y - y_pred) ** 2)
      # Number of observations
      n = len(y)
      # Number of parameters (including intercept)
      k = len(model.coef_) + (1 if model.fit_intercept else 0)  # Number of coefficients plus 1 for the intercept if present
      # Compute the AIC
      aic = n * np.log(rss / n) + 2 * k
      return aic


    def fit(self, x, y):
      exp_check = ExpressionChecker(x,y)
      x = exp_check.output()
      self.x = x
      self.linear_with_constant.fit(x, y)
      self.linear_without_constant.fit(x, y)
      aic_1 = self.calculate_aic(self.linear_with_constant, x, y)
      aic_2 = self.calculate_aic(self.linear_without_constant, x, y)
      if aic_1<aic_2:
         self.model = self.linear_with_constant
      else:
         self.model = self.linear_without_constant
      return self

    def predict(self, x):
      return self.model.predict(x)

    def print_eqn(self):
        # Extract the coefficients and intercept
        coeffs = self.model.coef_ 
        feature_names = self.x.columns()
        if self.model.fit_intercept:
          intercept = self.model.intercept_
          equation = f"y = {intercept:.3f}"
          equation += f" + ({coeffs:.3f}) * {feature_names}"
        else:
           equation += f" + ({coeffs:.3f}) * {feature_names}"
        print(equation)