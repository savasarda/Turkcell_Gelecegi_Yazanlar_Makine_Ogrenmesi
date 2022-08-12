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
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from sklearn import neighbors
from sklearn.svm import SVC
from catboost import CatBoostRegressor,CatBoostClassifier
from lightgbm import LGBMRegressor,LGBMClassifier
from xgboost import XGBRegressor,XGBClassifier
import warnings

df = pd.read_csv("diabetes.csv") 
y = df["Outcome"]
x= df.drop(["Outcome"],axis=1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)

modeller = [KNeighborsClassifier,
            LogisticRegression,
            SVC,
            MLPClassifier,
            DecisionTreeClassifier,
            RandomForestClassifier,
            GradientBoostingClassifier,
            CatBoostClassifier,
            LGBMClassifier,
            XGBClassifier]

sonuc = []
sonuclar = pd.DataFrame(columns=["Modeller","Accuracy"])

for model in modeller:
    isimler = model.__name__
    fitting = model().fit(x_train,y_train)
    y_pred = fitting.predict(x_test)
    dogruluk = accuracy_score(y_test,y_pred)
    sonuc = pd.DataFrame([[isimler,dogruluk*100]],columns=["Modeller","Accuracy"])
    sonuclar = sonuclar.append(sonuc)

sns.barplot(x="Accuracy",y="Modeller",data=sonuclar,color="r")
plt.xlabel("Accuracy %")
plt.title("Modellerin Doğruluk Oranları")


sonuclar.sort_values(["Accuracy"],ascending=False)