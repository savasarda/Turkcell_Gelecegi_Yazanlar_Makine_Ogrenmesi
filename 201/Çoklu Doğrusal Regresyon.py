from random import random
import pandas as pd
import numpy as np

df = pd.read_csv("Advertising.csv")
df.head()
df.drop("Unnamed: 0",axis=1,inplace=True)
x = df.drop('sales',axis=1)
y = df[["sales"]]

"""
#Statsmodels ile model kurmak
import statsmodels.api as sm
lm = sm.OLS(y,x) #önce bağımsız sonra bağımlı değişken
model = lm.fit() #yukarda değişkenleri yazdığmız için tekrar yazmaya gerek yok

model.summary()
"""

#scikit learn ile model oluşturma
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model = lm.fit(x,y) #önce bağımsız soran bağımlı değişkenler yazılır.
model.intercept_
model.coef_

#Tahmin
model.predict(x)
yeni_veri = [[230],[37.8],[69.2]]
yeni_veri = pd.DataFrame(yeni_veri).T #Transpose alınması lazım
model.predict(yeni_veri)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y,model.predict(x))
rmse = np.sqrt(mse)

#Model Tuning (Model Doğrulama)

#sınama seti --- hangi 80e 20 yi(yani en iyi durumu) alacagımızı bilemeyız bu yuzden k-katlı cv kullanıcaz.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20, random_state=1)

lm = LinearRegression()
model = lm.fit(x_train,y_train) #önce bağımsız, sonra bağımlı
np.sqrt(mean_squared_error(y_train,model.predict(x_train))) #eğitim hatası
np.sqrt(mean_squared_error(y_test,model.predict(x_test))) #test hatası

#k-katlı cv
from sklearn.model_selection import cross_val_score
cross_val_score(model,x_train,y_train, cv=10,scoring="neg_mean_squared_error")
#cv ile elde edilmiş mse
np.mean(-cross_val_score(model,x_train,y_train, cv=10,scoring="neg_mean_squared_error"))
#cv ile elde edilmiş rmse
np.sqrt(np.mean(-cross_val_score(model,x_train,y_train, cv=10,scoring="neg_mean_squared_error")))
#cv ile hesaplanan hata hesaplaması doğrulanmış bir hatadır.


