from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import BayesianRidge, LinearRegression
import seaborn as sns
import pandas as pd
import numpy as np
import pickle

class BayesianRegression:
    def __init__(self):
        self.full_data = None
        self.X_post = None
        self.y_pred = None
        self.y_std = None
        self.brr_poly = None
        
    def fit(self, X, y, degree):

        self.full_data = pd.DataFrame({"Skin color nuance": X, "Performance": y})
        X = X.values.reshape((-1, 1))

        self.brr_poly = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            StandardScaler(),
            BayesianRidge(),
        ).fit(X, y)

    def predict(self, start, stop, num_data):
        X_post = np.linspace(start, stop, num_data)
        X_post = X_post.reshape((-1, 1))
        y_pred, y_std = self.brr_poly.predict(X_post, return_std=True)
        self.X_post = X_post
        self.y_pred = y_pred
        self.y_std = y_std
        return self.y_pred, self.y_std

    def display(self, ax, model_name):
        sns.scatterplot(
            data=self.full_data, x="Skin color nuance", y="Performance", color="grey", alpha=0.5, ax=ax, label="Observation")
        ax.plot(self.X_post, self.y_pred, color="blue", label="Estimation")
        ax.fill_between(self.X_post.ravel(),
            self.y_pred - self.y_std,
            self.y_pred + self.y_std,
            color="lightblue", alpha=0.3)
        ax.axhline(self.full_data["Performance"].mean(), color="red", linestyle="--", label="Zero Difference")
        ax.set_title(model_name)
        ax.legend(loc="lower left")
        return ax

    def save_model(self, filepath):
        if self.brr_poly is not None:
            with open(filepath, "wb") as f:
                pickle.dump(self.brr_poly, f)
                print(f"Modle saved")
        

