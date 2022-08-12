from statistics import mean
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import cross_val_score

df = pd.read_csv("Hitters.csv")
df = df.dropna()
dms = pd.get_dummies(df[["League","Division","NewLeague"]])

y = df["Salary"]
x_ = df.drop(["Salary","League","Division","NewLeague",],axis=1).astype("float64")
x = pd.concat([x_, dms[["League_N", "Division_W", "NewLeague_N"]]],axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=42)

#Model Kurma
lasso_model = Lasso().fit(x_train,y_train)
lasso_model.intercept_
lasso_model.coef_

#farklı lambda değerlerine karşılık katsayılar
lasso = Lasso()
coefs = []
alphas = np.random.randint(0,1000,100)
for a in alphas:
    lasso.set_params(alpha = a)
    lasso.fit(x_train,y_train)
    coefs.append(lasso.coef_)

ax = plt.gca()
ax.plot(alphas,coefs)
ax.set_xscale("log")
#not: lassonun ridge den farkı şuydu : eğer katsayı yeteri kadar büyükse lambda 0 olabiliyordu.

#Tahmin
lasso_model.predict(x_train)[0:5]
lasso_model.predict(x_test)[0:5]
y_pred = lasso_model.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred))
r2_score(y_test, y_pred)

#Model Tuning
#Optimum lambda değerini bulmaya çalşıyoruz
lass_cv_model = LassoCV(alphas = alphas, cv=10,max_iter=100000).fit(x_train,y_train)
lass_cv_model.alpha_

lasso_tuned = Lasso().set_params(alpha=lass_cv_model.alpha_).fit(x_train,y_train)

y_pred = lasso_tuned.predict(x_test)
np.sqrt(mean_squared_error(y_test,y_pred))

pd.Series(lasso_tuned.coef_,index=x_train.columns)




