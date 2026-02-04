'''
In this Project we are finding the Heart Disease prediction using regression models
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import sklearn
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import pickle

from log_code import setup_logging
logger = setup_logging('main')

from handling_missing import RSI_tecnique
from variable_transformation import VARIABLE_TRANSFORMATION
from balance_data import balancing_data
from model_training import common
from sklearn.preprocessing import StandardScaler







class HEART_DISEASE_PREDICTION:
        def __init__(self,path):
            try:
                self.path = path
                self.df = pd.read_csv(self.path)
                logger.info(f'{self.df.columns} -> {self.df.shape}')
                #logger.info(f'Columns and datatypes:{self.df.info()}')
                self.X = self.df.iloc[:,:-1]
                self.y = self.df.iloc[:, -1]
                logger.info(f'X - Shape and columns: {self.X.shape} -> {self.X.columns}')
                logger.info(f'y - Shape and columns: {self.y.shape} ')

                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 42)
                logger.info(f'{self.X_train.columns}')
                #logger.info(self.y_train.info())

                logger.info(f'{self.X_train.head(5)}')
                logger.info(f'{self.y_train.head(5)}')

                logger.info(f'Training data size : {self.X_train.shape}')
                logger.info(f'Testing data size : {self.X_test.shape}')
                logger.info(f'Total no.of null values in data:{self.df.isnull().sum()}')

                logger.info(f'========================================')
            except Exception as e:
                error_type, error_msg, error_line = sys.exc_info()
                logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

        def missing_values(self):
            try:
                if self.X_train.isnull().sum().all() > 0 or self.X_test.isnull().sum().all() > 0:
                    self.X_train,self.X_test = RSI_tecnique.random_sample_imputataion(self.X_train, self.X_test)
                else:
                    logger.info(f'There are no null values in data:{self.X_train.isnull().sum()}')
            except Exception as e:
                error_type, error_msg, error_line = sys.exc_info()
                logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

        def var_transformation(self):
            try:
                for i in self.X_train.columns:
                    logger.info(f'{self.X_train[i].dtype}')
                logger.info(f'{self.X_train.columns}')
                logger.info(f'{self.X_test.columns}')
                self.X_train, self.X_test = VARIABLE_TRANSFORMATION.variable_trans(self.X_train, self.X_test)
                logger.info(f'{self.X_train.columns} --> {self.X_train.shape}')
                logger.info(f'{self.X_test.columns} --> {self.X_test.shape}')

                logger.info(f'=================================================================')
            except Exception as e:
                error_type, error_msg, error_line = sys.exc_info()
                logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

        def balancing_data(self):
            try:
                logger.info(f"Before SMOTE - Class Distribution: {self.y_train.value_counts().to_dict()}")
                self.X_train, self.y_train = balancing_data(self.X_train, self.y_train)
                logger.info(f"After SMOTE - Class Distribution: {self.y_train.value_counts().to_dict()}")
                logger.info(f"Balanced X_train shape: {self.X_train.shape}")
                logger.info(f"Balanced y_train shape: {self.y_train.shape}")

            except Exception as e:
                error_type, error_msg, error_line = sys.exc_info()
                logger.info(f'Error in Line no : {error_line.tb_lineno} : due to {error_msg}')

        def scaling(self):
            try:
                logger.info(f'{self.X_train.shape}')
                logger.info(f'{self.X_test.shape}')
                logger.info(f'Before \n:{self.X_train}')
                logger.info(f'Before \n:{self.X_test}')
                scale_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

                sc = StandardScaler()
                sc.fit(self.X_train[scale_cols])

                self.X_train[scale_cols] = sc.transform(self.X_train[scale_cols])
                self.X_test[scale_cols] = sc.transform(self.X_test[scale_cols])

                # Save scaler for inference
                with open('scaler.pkl', 'wb') as f:
                    pickle.dump(sc, f)

                logger.info(f'{self.X_train.shape}')
                logger.info(f'{self.X_test.shape}')
                logger.info(f'Before \n:{self.X_train}')
                logger.info(f'Before \n:{self.X_test}')

            except Exception as e:
                error_type, error_msg, error_line = sys.exc_info()
                logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

        def models_training(self):
            try:
                logger.info(f'Training Started')
                common(self.X_train, self.y_train, self.X_test, self.y_test)
                logger.info(f'Training Completed')
                logger.info("Training Logistic Regression ")
            except Exception as e:
                error_type, error_msg, error_line = sys.exc_info()
                logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')
        def final_model(self):
            try:
                models_training()
            except Exception as e:
                error_type, error_msg, error_line = sys.exc_info()
                logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')


if __name__ == '__main__':
    try:
        obj = HEART_DISEASE_PREDICTION(f'C:\\Users\\VARSHINI\\Downloads\\Heart_disease_prediction\\heart_disease.csv')
        obj.missing_values()
        obj.var_transformation()
        obj.balancing_data()
        obj.scaling()
        obj.models_training()

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')