import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score


df = pd.read_csv("Hitters.csv")

df = df.dropna()
dms = pd.get_dummies(df[["League", "Division", "NewLeague"]])

y = df["Salary"]
x_ = df.drop(["Salary", "League", "Division", "NewLeague"], axis=1).astype("float64")
x = pd.concat([x_, dms[["League_N", "Division_W", "NewLeague_N"]]],axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=42)
df.shape

ridge_model = Ridge(alpha = 0.1).fit(x_train,y_train)
ridge_model.coef_
ridge_model.intercept_

lambdalar = 10**np.linspace(10,-2,100)*0.5

ridge_model = Ridge()
katsayılar = []

for i in lambdalar:
    ridge_model.set_params(alpha=i)
    ridge_model.fit(x_train,y_train)
    katsayılar.append(ridge_model.coef_)

katsayılar
"""
ax = plt.gca()
ax.plot(lambdalar,katsayılar)
ax.set_xscale("log")
"""
#Tahmin
y_pred = ridge_model.predict(x_train)

#train hatası
rmse = np.sqrt(mean_squared_error(y_train,y_pred))
rmse

np.sqrt(np.mean(-cross_val_score(ridge_model,x_train,y_train, cv=10,scoring="neg_mean_squared_error")))

#test hatası
y_pred = ridge_model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
rmse

#Model Tuning(Ayarlama)
lambdalar1 = np.random.randint(0,1000,100)
lambdalar2 = 10**np.linspace(10,-2,100)*0.5
ridgecv= RidgeCV(alphas = lambdalar2, scoring = "neg_mean_squared_error", cv = 10, normalize=True)
ridgecv.fit(x_train,y_train)
ridgecv.alpha_

#final modeli
ridge_tuned = Ridge(alpha = ridgecv.alpha_).fit(x_train, y_train)
y_pred = ridge_tuned.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))





