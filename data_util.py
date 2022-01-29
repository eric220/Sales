import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def label_encode(df):
    le = LabelEncoder()
    df.qtr = le.fit_transform(df.qtr)
    df.year = le.fit_transform(df.year)
    return df

def data_prep(df):
    dummies = pd.get_dummies(df.Day)
    df = pd.concat([df, dummies], axis = 1)
    df['rol_mean'] = df.Weekly_Sales.rolling(window=3).mean()
    df.dropna(inplace = True)
    return df

def get_split(df, holdout):
    features = df.drop(['Weekly_Sales', 'Store', 'Date', 'Day'], axis = 1)
    labels = df.Weekly_Sales
    holdout = holdout                
    X_train, X_test, y_train, y_test = features[:-holdout].copy(), features[-holdout:].copy(),labels[:-holdout].copy(), labels[-holdout:].copy()
    return X_train, X_test, y_train, y_test

def get_x_y_split(df, keep_out):
    df = data_prep(df)
    df = label_encode(df)
    return get_split(df, keep_out)