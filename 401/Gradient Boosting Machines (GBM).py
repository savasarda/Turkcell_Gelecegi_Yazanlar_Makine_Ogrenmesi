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
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import warnings

df = pd.read_csv("diabetes.csv") 
y = df["Outcome"]
x= df.drop(["Outcome"],axis=1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)

#Model&Tahmin
gbm_model = GradientBoostingClassifier().fit(x_train,y_train)
y_pred = gbm_model.predict(x_test)
accuracy_score(y_test,y_pred) #yüzde75

#Model Tuning
gbm_params = {"learning_rate" : [0.1,0.01,0.001,0.05],
              "n_estimators" : [100,300,500,1000],
              "max_depth" : [2,3,5,8]}

gbm_cv_model = GridSearchCV(gbm_model,gbm_params,cv=10,n_jobs=-1,verbose=2).fit(x_train,y_train)
gbm_cv_model.best_params_

#Final Model
gbm_tuned_model = GradientBoostingClassifier(learning_rate=0.01,n_estimators=500,max_depth=5).fit(x_train,y_train)
y_pred = gbm_tuned_model.predict(x_test)
accuracy_score(y_test,y_pred) #yüzde73

#Değişken Önem Düzeyleri
feature_imp = pd.Series(gbm_tuned_model.feature_importances_,index=x_train.columns).sort_values(ascending=False)
sns.barplot(x=feature_imp,y=feature_imp.index)
plt.xlabel("Değişken Önem Skorları")
plt.ylabel("Değişkenler")
plt.title("Değişken Önem Düzeyleri")
plt.show()
