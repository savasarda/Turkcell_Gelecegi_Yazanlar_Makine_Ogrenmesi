import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import cross_val_score

df = pd.read_csv("Hitters.csv")
df = df.dropna()
dms = pd.get_dummies(df[["League","Division","NewLeague"]])

y = df["Salary"]
x_ = df.drop(["Salary","League","Division","NewLeague",],axis=1).astype("float64")
x = pd.concat([x_, dms[["League_N", "Division_W", "NewLeague_N"]]],axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=42)

enet_model = ElasticNet().fit(x_train,y_train)
enet_model.coef_
enet_model.intercept_

enet_model.predict(x_train)[0:10]
enet_model.predict(x_test)[0:10]
y_pred = enet_model.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred))
r2_score(y_test,y_pred)

#Model Tuning
alphas = np.random.randint(0,1000,100)
enet_cv_model = ElasticNetCV(alphas=alphas ,cv=10).fit(x_train,y_train)
enet_cv_model.alpha_
enet_cv_model.intercept_
enet_cv_model.coef_

#final modeli
enet_tuned = ElasticNet(alpha=enet_cv_model.alpha_).fit(x_train,y_train)
y_pred = enet_tuned.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred))
