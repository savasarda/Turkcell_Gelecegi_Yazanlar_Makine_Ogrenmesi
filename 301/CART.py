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
x_train = pd.DataFrame(x_train["Hits"])
x_test = pd.DataFrame(x_test["Hits"])

cart_model = DecisionTreeRegressor(max_leaf_nodes=10).fit(x_train,y_train)

x_grid = np.arange(min(np.array(x_train)),max(np.array(x_train)),0.01)
x_grid = x_grid.reshape((len(x_grid)),1)

plt.scatter(x_train,y_train,color= "red")
plt.plot(x_grid,cart_model.predict(x_grid),color="blue")

plt.title("CART REGRESYON AĞACI")
plt.xlabel("Atış Sayısı(Hits)")
plt.ylabel("Maaş(Salary)")

#tek değişkenli Tahmin
y_pred = cart_model.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred))

#tüm değişkenli tahmin
df = pd.read_csv("Hitters.csv")

df = df.dropna()
dms = pd.get_dummies(df[["League", "Division", "NewLeague"]])

y = df["Salary"]
x_ = df.drop(["Salary", "League", "Division", "NewLeague"], axis=1).astype("float64")
x = pd.concat([x_, dms[["League_N", "Division_W", "NewLeague_N"]]],axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=42)

cart_model = DecisionTreeRegressor().fit(x_train,y_train)
y_pred = cart_model.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred))

#Model Tuning
cart_params = {"max_depth": [2,3,4,5,10,20],
               "min_samples_split" : [2,10,5,30,50,100]}

cart_model = DecisionTreeRegressor()
cart_cv_model = GridSearchCV(cart_model,cart_params, cv=10).fit(x_train,y_train) 
cart_cv_model.best_params_

#final model
cart_tuned_model = DecisionTreeRegressor(max_depth=5, min_samples_split=50).fit(x_train,y_train)
y_pred = cart_tuned_model.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred))

