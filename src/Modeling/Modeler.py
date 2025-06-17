# === Required Libraries ===
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings("ignore")

class ClaimsModelPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_data = None
        self.filtered_data = None
        self.prepared_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def read_and_prepare(self):
        # Load dataset and filter claims with values > 0
        self.raw_data = pd.read_csv(self.data_path, sep="|")
        self.filtered_data = self.raw_data[self.raw_data['TotalClaims'] > 0].copy()

        # Convert CapitalOutstanding to numeric, discard rows with missing values
        self.filtered_data['CapitalOutstanding'] = pd.to_numeric(
            self.filtered_data['CapitalOutstanding'], errors='coerce'
        )
        self.filtered_data.dropna(subset=['TotalClaims', 'CapitalOutstanding'], inplace=True)

    def build_features(self):
        # Define predictors
        selected_columns = [
            'Cubiccapacity', 'Kilowatts', 'CapitalOutstanding', 'SumInsured',
            'NewVehicle', 'Gender', 'Province', 'VehicleType'
        ]
        # Keep necessary columns and handle nulls
        self.prepared_data = self.filtered_data[selected_columns + ['TotalClaims']].dropna()

        # Encode categorical variables using one-hot encoding
        self.prepared_data = pd.get_dummies(
            self.prepared_data,
            columns=['NewVehicle', 'Gender', 'Province', 'VehicleType'],
            drop_first=True
        )

    def split_dataset(self):
        # Separate target from features and perform train-test split
        X = self.prepared_data.drop('TotalClaims', axis=1)
        y = self.prepared_data['TotalClaims']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def evaluate_model(self, model, label):
        # Fit and evaluate the model
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        rmse_score = np.sqrt(mean_squared_error(self.y_test, predictions))
        r2_val = r2_score(self.y_test, predictions)
        print(f"{label} — RMSE: {rmse_score:.2f}, R²: {r2_val:.4f}")
        return model

    def train_all_models(self):
        # Train different regression models and evaluate them
        linear = self.evaluate_model(LinearRegression(), "Linear Regression")
        forest = self.evaluate_model(RandomForestRegressor(random_state=42), "Random Forest")
        booster = self.evaluate_model(XGBRegressor(random_state=42, n_jobs=-1), "XGBoost")
        return linear, forest, booster

    def interpret_model(self, trained_model):
        # Use SHAP to interpret the model
        explainer = shap.Explainer(trained_model)
        shap_values = explainer(self.X_test)
        shap.plots.beeswarm(shap_values)
        plt.show()
