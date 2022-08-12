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
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
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
cart_model = DecisionTreeClassifier().fit(x_train,y_train)
y_pred = cart_model.predict(x_test)
accuracy_score(y_test,y_pred) #yüzde70

#Model Tuning
cart_params = {"max_depth" :[2,3,5,8,10],
               "min_samples_split" : [1,3,5,10,20,50]}
cart_cv_params = GridSearchCV(cart_model,cart_params,cv=10,n_jobs=-1,verbose=2).fit(x_train,y_train)
cart_cv_params.best_params_

#Final Model
cart_tuned_models = DecisionTreeClassifier(max_depth=5,min_samples_split=20).fit(x_train,y_train)
y_pred = cart_tuned_models.predict(x_test)
accuracy_score(y_test,y_pred) #yüzde75