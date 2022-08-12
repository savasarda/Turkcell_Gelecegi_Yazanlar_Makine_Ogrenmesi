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

gbm_model = GradientBoostingRegressor().fit(x_train,y_train)
y_pred = gbm_model.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred))

#Model Tuning
gbm_params = {"learning_rate": [0.001,0.1,0.01],
              "max_depth": [3,5,8],
              "n_estimators": [100,200,500],
              "subsample": [1,0.5,0.8],
              "loss": ["ls","lad","quantile"]}

gbm_cv_model = GridSearchCV(gbm_model,gbm_params,cv=10,n_jobs=-1,verbose=2).fit(x_train,y_train)
gbm_cv_model.best_params_

#final model
gbm_model_tuned = GradientBoostingRegressor(learning_rate=0.1,max_depth=3,loss="lad",n_estimators=200,subsample=1).fit(x_train,y_train)
y_pred = gbm_model_tuned.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred))

#Değişken Önem Düzeyi
Importance = pd.DataFrame({"Importance":gbm_model_tuned.feature_importances_*100},
            index=x_train.columns)

Importance.sort_values(by = "Importance",axis=0,ascending=True).plot(kind="barh",color="r")

plt.xlabel("Variable Importance")
plt.gca().legend_ =None
