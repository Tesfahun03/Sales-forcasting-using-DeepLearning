import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class MergeData:
    """a class for merging to dataset into single dataset by using store as a key
    """

    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def merge_and_load(self):
        merged_df = pd.merge(self.dataset1, self.dataset2,
                             on="Store", how="left")
        merged_df.to_csv('../data/merged_data.csv')
        df = pd.read_csv(
            '../data/merged_data.csv', parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        return df


class DataCleaner:
    def __init__(self, data):
        self.data = data
    try:
        def fillna_with_previos_value(self, columns: list):
            """filling NA values with previus value in the dataset

            Args:
                columns (list): _dataseries for a datframe_

            Returns:
                _PD.DataFrame_: _new dataframe with value filled_
            """
            logging.info(' filling Na values with previos one started...')
            for col in columns:
                self.data[col] = self.data[col].fillna(method='ffill')

            logging.info('filling Na values completed. ')
            return self.data
    except Exception as e:
        logging.info(f'error occuered during filling values {e}')

    def plot_destribution(self, column: pd.Series, kind):
        """plots a graph for a specific column

        Args:
            column (_pd.Series_): pandas series or column of a dataset
            kind (_plot Kind_): _plot kind[bar, barh, line, hist,..]_

        Returns:
            _Axes_: _plots a graph of type your kind_
        """

        logging.info(f"plotting {kind} graph for {column} ")
        try:
            return self.data[column].value_counts().plot(kind=kind)
        except KeyError as K:
            logging(
                f'column name :| {column} | does not exist the Dataframe. ')
