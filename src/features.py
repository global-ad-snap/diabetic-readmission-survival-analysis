import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def standardize_missing_values(df: pd.DataFrame):
    return df.replace({"?": np.nan, "Unknown/Invalid": np.nan})

def basic_casting(df: pd.DataFrame):
    # cast numeric + diagnosis codes
    ...
    return df

def make_survival_labels(df: pd.DataFrame, simulate_time: bool=False, seed: int=42):
    ...
    return df

def prepare_cox_dataset(df: pd.DataFrame, drop_cols: list=None, scaler=None):
    ...
    return df_cox, scaler 
