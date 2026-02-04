import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from log_code import setup_logging
logger = setup_logging('balance_data')

from imblearn.over_sampling import SMOTE

def balancing_data(X_train, y_train):
    try:
        y_train = y_train.values.ravel()
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        return X_res, y_res
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.error(f'Error in line {error_line.tb_lineno}: {error_msg}')