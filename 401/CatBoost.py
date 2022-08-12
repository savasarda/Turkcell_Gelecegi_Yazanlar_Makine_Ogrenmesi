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

#Model&Tahmin
catb_model = CatBoostClassifier().fit(x_train,y_train,verbose=False)
y_pred = catb_model.predict(x_test)
accuracy_score(y_test,y_pred)#yüzde74

#Model Tuning
catb_model = CatBoostClassifier(verbose=False)
catb_params = {"iterations" : [200,500,1000],
               "learning_rate" : [0.01,0.03,0.1],
               "depth" : [4,5,8]}
#catb_cv_model = GridSearchCV(catb_model,catb_params,cv=5,n_jobs=-1,verbose=2).fit(x_train,y_train)
#catb_cv_model.best_params_

#Final Model
catb_tuned_model= CatBoostClassifier(iterations=200,depth=8,learning_rate=0.01,verbose=False).fit(x_train,y_train)
y_pred = catb_tuned_model.predict(x_test)
accuracy_score(y_test,y_pred) #yüzde75

#Değişken Önem Düzeyleri
feature_imp = pd.Series(catb_tuned_model.feature_importances_,index=x_train.columns).sort_values(ascending=False)
sns.barplot(x=feature_imp,y=feature_imp.index)
plt.xlabel("Değişken Önem Skorları")
plt.ylabel("Değişkenler")
plt.title("Değişken Önem Düzeyleri")
plt.show()