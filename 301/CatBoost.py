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

from warnings import filterwarnings
filterwarnings("ignore")

#Model&Tahmin
df = pd.read_csv("Hitters.csv")

df = df.dropna()
dms = pd.get_dummies(df[["League", "Division", "NewLeague"]])

y = df["Salary"]
x_ = df.drop(["Salary", "League", "Division", "NewLeague"], axis=1).astype("float64")
x = pd.concat([x_, dms[["League_N", "Division_W", "NewLeague_N"]]],axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=42)

from catboost import CatBoostRegressor
catb_model = CatBoostRegressor().fit(x_train,y_train)
y_pred = catb_model.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred))

#Model Tuning
catb_params = {"iterations": [200,500,1000],
               "learning_rate": [0.01,0.1],
               "depth": [3,6,8]}

catb_cv_model = GridSearchCV(catb_model,catb_params,cv=5,n_jobs=-1,verbose=2).fit(x_train,y_train)
catb_cv_model.best_params_

#final model
catb_tuned_model = CatBoostRegressor(depth=3,iterations=500,learning_rate=0.1).fit(x_train,y_train)
y_pred = catb_tuned_model.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred))

#EN DÜŞÜK HATAYI CATBOOSTLA BULMUŞ OLDUK.