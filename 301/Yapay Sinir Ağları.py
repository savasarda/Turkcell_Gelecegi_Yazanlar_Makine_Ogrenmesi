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

#Model&Tahmin

scaler = StandardScaler()
scaler.fit(x_train,y_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

mlp_model = MLPRegressor().fit(x_train_scaled,y_train)
y_pred = mlp_model.predict(x_test_scaled)
np.sqrt(mean_squared_error(y_test,y_pred))

#Model Tuning
mlp_params = {"alpha":[0.1, 0.01, 0.02, 0.001, 0.0001],
              "hidden_layer_sizes": [(10,20), (5,5), (100,100)]}

mlp_cv_model = GridSearchCV(mlp_model, mlp_params, cv=5,verbose=2,n_jobs=-1).fit(x_train_scaled,y_train)  #işlemi hızlandırmak için yazılır.n_jobs = -1 sayesinde bütün işlemciler kullanılr.
mlp_cv_model.best_params_

#final model
mlp_tuned = MLPRegressor(alpha=0.01,hidden_layer_sizes=(100,100)).fit(x_train_scaled,y_train)
y_pred = mlp_tuned.predict(x_test_scaled)
np.sqrt(mean_squared_error(y_test,y_pred))

#675 ten 357 e düştü hata tuning yaptığımızda.
