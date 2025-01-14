import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')



class SplitData:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    logging.info('spliting the dataset into 80 and 20% ')
    def split_data(self):
        """_splits the data into 80% of training data and 20% of testing data_

        Returns:
            _train and test data_: _training and testing datas_
        """
        return train_test_split(self.x, self.y, test_size=0.2, random_state=42)


class TrainData:
    """_a class for training multiple machine learning algorithm for the same training and testing sets._
    """

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
  
    logging.info('training begins with the training dataset')
    def random_forest(self):
        """_initialize the Random forest  model and fit the training and testing set from the dataset_

        Returns:
            RandomForestRegressor()_: _a Random Forest  model on the training data_
        """
        random_forest_model = RandomForestRegressor(
            n_estimators=100, n_jobs=-1)  # Use all CPU cores
        random_forest_model.fit(self.x_train, self.y_train)
        logging.info('Training randomforest regressor ends')
        return random_forest_model

    def xgboost(self):
        """_initialize the XG BOOST model and fit the training and testing set from the dataset_

        Returns:
            XGBOOST()_: _a XGBoost model on the training data_
        """
        logging.info('training XGB Regressor is now begin')
        xg_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
        xg_model.fit(self.x_train, self.y_train)
        logging.info('Training XGB Regressor regressor ends')
        return xg_model


class EvaluateModel:
    """_class for evualuating the accuracy of a given model_
    """
    logging.info('Evaluating models now begin..')
    def evaluate_model(self, model, x_test, y_test):
        """_evaluates the errors of the model using accuracy metrics_

        Args:
            model (_[LinearRegression, DecisionTreeRegressor, RandomForestRegressor, XGBOOST]_): _regression models to measure thier accuracy_
            x_test (_pd.DataFrame_): _Pandas dataframe of testing data for the features columns_
            y_test (_pd.DataFrame_): _pandas dataframe of testing data for the target column_

        Returns:
            _Accuracy metrics_: _' '_
        """
        y_pred = model.predict(x_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return mae, mse, r2, y_pred
