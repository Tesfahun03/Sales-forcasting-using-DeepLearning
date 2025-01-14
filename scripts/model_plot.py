import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class PlotMetrics:
    """_plot the acuuracy metrics for each_
    """

    def __init__(self, models, mae_scores, mse_scores, r2_scores):
        self.models = models
        self.mae_scores = mae_scores
        self.mse_scores = mse_scores
        self.r2_scores = r2_scores

    def plot(self):
        """_plot the acuuracy metrics for mean absolute error, mean squared error and r2-score_
        """
        logging.info('plotting graph for accuracy metricses')
        plt.figure(figsize=(6, 4))
        plt.bar(self.models, self.mae_scores, color='green')
        plt.xlabel('models')
        plt.ylabel('Mean absolute error ')
        plt.title('comparison of MAE scores')
        plt.xticks(rotation=45)
        plt.show()

        # plot for mean squared error
        logging.info('plotting graph for mean squared error')
        plt.figure(figsize=(6, 4))
        plt.bar(self.models, self.mse_scores, color='yellow')
        plt.xlabel('models')
        plt.ylabel('Mean squared error ')
        plt.title('comparison of MsE scores')
        plt.xticks(rotation=45)
        plt.show()

        # plot for r2-scores
        logging.info('plotting graph for r2-scores')
        plt.figure(figsize=(6, 4))
        plt.bar(self.models, self.r2_scores, color='red')
        plt.xlabel('models')
        plt.ylabel('r2_scores  ')
        plt.title('comparison of  r2-scores')
        plt.xticks(rotation=45)
        plt.show()
