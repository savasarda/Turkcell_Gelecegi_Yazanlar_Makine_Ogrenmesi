import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from warnings import filterwarnings


filterwarnings("ignore")

df = pd.read_csv("Hitters.csv")
df = df.dropna()
dms = pd.get_dummies(df[["League", "Division", "NewLeague"]])

def compML(df,y,alg):
    #train-test ayrımı
    y = df[y]
    x_ = df.drop(["Salary", "League", "Division", "NewLeague"], axis=1).astype("float64")
    x = pd.concat([x_, dms[["League_N", "Division_W", "NewLeague_N"]]],axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=42)
    #modelleme
    model = alg().fit(x_train,y_train)
    y_pred = model.predict(x_test)
    RMSE = np.sqrt(mean_squared_error(y_test,y_pred))
    model_ismi = alg.__name__
    print(model_ismi,"Modeli için Test Hatası:",RMSE)


models = [LGBMRegressor,
          XGBRegressor,
          GradientBoostingRegressor,
          RandomForestRegressor,
          DecisionTreeRegressor,
          MLPRegressor,
          KNeighborsRegressor,
          SVR,
          CatBoostRegressor]

for i in models:
    compML(df,"Salary",i)




