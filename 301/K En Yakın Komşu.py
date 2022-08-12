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

#KNN
df = pd.read_csv("Hitters.csv")

df = df.dropna()
dms = pd.get_dummies(df[["League", "Division", "NewLeague"]])

y = df["Salary"]
x_ = df.drop(["Salary", "League", "Division", "NewLeague"], axis=1).astype("float64")
x = pd.concat([x_, dms[["League_N", "Division_W", "NewLeague_N"]]],axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=42)

#Model
knn_model = KNeighborsRegressor().fit(x_train,y_train)
knn_model.n_neighbors

dir(knn_model)

#Tahmin
y_pred = knn_model.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred))

#Model Tuning

RMSE = []

for k in range(10):
    k = k+1
    knn_model= KNeighborsRegressor(n_neighbors=k).fit(x_train,y_train)
    y_pred = knn_model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    RMSE.append(rmse)
    print("k =", k, "için RMSE değeri:", rmse)

#GridSearchCV -- HİPERPARAMETRELERİN PARAMETRELERİNİ BELİRLER.
knn_params = {"n_neighbors": np.arange(1,30,1)}
knn = KNeighborsRegressor()
knn_cv_model = GridSearchCV(knn, knn_params, cv=10).fit(x_train,y_train)
knn_cv_model.best_params_ #az önce elle yaptığımız şeyi GridSearchCV ile bulduk. {'n_neighbors': 8}

#final_model
knn_tuned = KNeighborsRegressor(n_neighbors=knn_cv_model.best_params_["n_neighbors"]).fit(x_train,y_train)
y_pred = knn_tuned.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred))

