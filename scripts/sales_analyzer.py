import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from statsmodels.tsa.seasonal import seasonal_decompose

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class AnalyzeSales:
    def __init__(self, data):
        self.data = data

    def plot_holiday_effect(self):
        logging.info('plotting for holiday effects')

        self.data['IsHoliday'] = self.data['StateHoliday'] | (
            self.data.index.month == 12)
        holiday_effect = self.data.groupby('IsHoliday')['Sales'].mean()
        plt.figure(figsize=(6, 6))
        plt.pie(
            holiday_effect,
            labels=holiday_effect.index.map(
                {True: 'Holiday', False: 'Non-Holiday'}),
            autopct='%1.1f%%',
            startangle=140)
        # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.axis('equal')
        plt.title('Pie Chart for Holiday sales effect')

        plt.show()

    def plot_weekly_sales(self):
        logging.info("Plotting weekly sales...")
        weekly_sales = self.data['Sales'].resample('W').sum()
        plt.figure(figsize=(15, 7))
        plt.plot(weekly_sales.index, weekly_sales)
        plt.title('Weekly Sales Over Time')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.show()

    def plot_monthly_sales(self):
        logging.info("Plotting monthly sales...")
        monthly_sales = self.data['Sales'].resample('M').sum()
        plt.figure(figsize=(15, 7))
        plt.plot(monthly_sales.index, monthly_sales)
        plt.title('Monthly Sales Over Time')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.show()

    def seasonal_decomposition(self):
        logging.info("Performing seasonal decomposition...")
        monthly_sales = self.data['Sales'].resample('M').sum()
        result = seasonal_decompose(monthly_sales, model='additive')
        result.plot()
        plt.tight_layout()
        plt.show()

    def plot_promo_effect(self):
        logging.info("Plotting promo effect over time...")
        monthly_promo_sales = self.data.groupby(
            [self.data.index.to_period('M'), 'Promo'])['Sales'].mean().unstack()
        monthly_promo_sales.columns = ['No Promo', 'Promo']

        monthly_promo_sales[['No Promo', 'Promo']].plot(figsize=(15, 7))
        plt.title('Monthly Average Sales: Promo vs No Promo')
        plt.xlabel('Date')
        plt.ylabel('Average Sales')
        plt.legend(['No Promo', 'Promo'])
        plt.show()

    def plot_assortment_effect(self):
        logging.info("Plotting assortment effect over time...")
        monthly_assortment_sales = self.data.groupby(
            [self.data.index.to_period('M'), 'Assortment'])['Sales'].mean().unstack()
        monthly_assortment_sales.plot(figsize=(15, 7))
        plt.title('effect of monthly assortment on a sales')
        plt.xlabel('Assortment')
        plt.ylabel('Average Sales')
        plt.legend(['No Promo', 'Promo'])
        plt.show()

    def plot_sales_vs_customers(self):
        logging.info("Plotting sales vs customers scatter plot...")
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            self.data['Customers'], self.data['Sales'], c=self.data.index, cmap='viridis')
        plt.colorbar(scatter, label='Date')
        plt.title('Sales vs Customers Over Time')
        plt.xlabel('Number of Customers')
        plt.ylabel('Sales')
        plt.show()
