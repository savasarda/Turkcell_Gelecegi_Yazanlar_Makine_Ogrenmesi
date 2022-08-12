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

rf_model = RandomForestRegressor(random_state=42).fit(x_train,y_train)
y_pred = rf_model.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred))

#Model Tuning

rf_params = {"max_depth": [5,8,10],
             "max_features": [2,5,10],
             "n_estimators": [200,500,1000,2000],
             "min_samples_split": [2,10,80,100]}

rf_cv_model = GridSearchCV(rf_model,rf_params,cv=10, n_jobs=-1)
rf_cv_model.best_params_

rf_model_tuned = RandomForestRegressor(max_depth=8,max_features=2,min_samples_split=2,n_estimators=200).fit(x_train,y_train)
y_pred = rf_model_tuned.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred))

#Değişken Önem Düzeyi
Importance = pd.DataFrame({"Importance":rf_model_tuned.feature_importances_*100},
            index=x_train.columns)

Importance.sort_values(by = "Importance",axis=0,ascending=True).plot(kind="barh",color="r")

plt.xlabel("Variable Importance")
plt.gca().legend_ =None


