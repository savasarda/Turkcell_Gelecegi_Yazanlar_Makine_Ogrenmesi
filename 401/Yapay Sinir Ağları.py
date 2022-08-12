import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.neural_network import MLPRegressor,MLPClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVC
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import warnings

df = pd.read_csv("diabetes.csv") 
y = df["Outcome"]
x= df.drop(["Outcome"],axis=1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)

#Model&Tahmin
mlpc_model = MLPClassifier().fit(x_train,y_train)
y_pred = mlpc_model.predict(x_test)
accuracy_score(y_test,y_pred) #yüzde71

#Model Tuned
mlpc_params = {"alpha" : [1,5,0.1,0.01,0.03,0.005,0.0001],
               "hidden_layer_sizes" : [(10,10),(100,100,100),(100,100),(3,5)]}
mlpc_model = MLPClassifier(solver="lbfgs")
mlpc_cv_model = GridSearchCV(mlpc_model,mlpc_params,cv=10,n_jobs=-1,verbose=2).fit(x_train,y_train)
mlpc_cv_model.best_params_

#Final Model
mlpc_tuned_model = MLPClassifier(solver="lbfgs",alpha=5,hidden_layer_sizes=(3,5)).fit(x_train,y_train)
y_pred = mlpc_tuned_model.predict(x_test)
accuracy_score(y_test,y_pred) #yüzde77