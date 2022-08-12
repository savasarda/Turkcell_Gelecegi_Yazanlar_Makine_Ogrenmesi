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

svr_model = SVR(kernel="linear").fit(x_train,y_train)

svr_model.intercept_
svr_model.coef_

#test
y_pred = svr_model.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred))

#Model Tuning
svr_params = {"C": [0.1, 0.5, 1, 3]}
svr_cv_model = GridSearchCV(svr_model,svr_params,cv=5, verbose=2, n_jobs=-1).fit(x_train,y_train)
svr_cv_model.best_params_

svr_tuned = SVR(kernel="linear",C=0.5).fit(x_train,y_train)
y_pred = svr_tuned.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred))


