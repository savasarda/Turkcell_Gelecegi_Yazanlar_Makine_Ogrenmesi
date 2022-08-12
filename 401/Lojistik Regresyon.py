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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVC
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

df = pd.read_csv("diabetes.csv")
df.head()

#Model-Tahmin
df["Outcome"].value_counts()
df.describe().T

y = df["Outcome"]
x = df.drop(["Outcome"],axis=1)

log_model = LogisticRegression(solver="liblinear").fit(x,y)
log_model.intercept_
log_model.coef_

log_model.predict(x)[0:10]
y[0:10]

y_pred = log_model.predict(x)
confusion_matrix(y,y_pred)
accuracy_score(y,y_pred) #yüzde 77 başarılı tahmin yapılmış
classification_report(y,y_pred)

log_model.predict_proba(x)[0:10]  #olasılık değerleriyle tahmin yapar

logit_roc_auc = roc_auc_score(y,log_model.predict(x))
fpr,tpr,thresholds = roc_curve(y,log_model.predict_proba(x)[:,1])
plt.figure()
plt.plot(fpr,tpr,label="AUC (area = %0.2f)" % logit_roc_auc)
plt.plot([0,1],[0,1],"r--")
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel("False Positive Rate")
plt.xlabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.savefig("Log_ROC")
plt.show()

#Model Tuning
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)
log_model = LogisticRegression(solver="liblinear").fit(x_train,y_train)
y_pred = log_model.predict(x_test)
accuracy_score(y_test,y_pred) #yüzde 75 başarılı modelimiz

cross_val_score(log_model,x_test,y_test,cv=10).mean() #hiper parametrelerinin optimum değerlerine ulaşmak için 'cross_val_score' kullanıyoruz


