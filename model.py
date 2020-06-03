
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib
import numpy as np
import time
import pickle
import pandas as pd
import sklearn
from sklearn import linear_model

data = pd.read_csv('Resources/House_prices_train_FINAL.csv')
y = data['SalePrice']
x = data[['LotArea','TotalSF','WasRemodeled','IsNew','BedroomAbvGr','TotalBathrooms','OverallCond','ExterCond','HasBasement','HasGarage','HasPorch','HasPool','KitchenQual','FireplaceQu','GarageFinish','BsmtFinType2']]
x = x.apply(pd.to_numeric, errors='coerce')
y = y.apply(pd.to_numeric, errors='coerce')
x.fillna(0, inplace=True)
y.fillna(0, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.5, random_state = 42)

reglinear = linear_model.LinearRegression()
reglinear.fit(X_train,y_train)


pickle.dump(reglinear, open('model.pkl','wb'))


model = pickle.load(open('model.pkl','rb'))

