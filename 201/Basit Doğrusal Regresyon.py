import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("advertising.csv")
df = df.iloc[:,1:len(df)]
df.head()
df.info()

#sns.jointplot(x= "TV", y= "sales", data = df, kind = "reg")
 
#Modelleme
from sklearn.linear_model import LinearRegression
x = df[["TV"]]
y = df[["sales"]]

reg = LinearRegression()
model = reg.fit(x,y) 
str(model)  #pek önemli birşey değil
model.intercept_ #sabitimiz = b0
model.coef_ #b1 katsaysısı
model.score(x,y) #rkare = bağımlı değişkendeki değişikliğin, bağımsız değişkence açıklanma yüzdesidir.yani satıştaki değişikliğin yüzde 61 i TV ile açıklanmaktadır.

#Tahmin
g = sns.regplot(x= df["TV"], y= df["sales"])
g.set_title("Model Denklemi: Sales = 7.03 + TV*0,047")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10,310)
plt.ylim(bottom=0)

model.predict(x)
yeni_veri = [[5],[15],[30]]
model.predict(yeni_veri)

#MSE - RMSE
gercek_satis = y
tahmin_edilen_satis = pd.DataFrame(model.predict(x))  #model.predict(x) array halinde, onu dataframe çevirmemiz lazım.
hatalar = pd.concat([gercek_satis,tahmin_edilen_satis],axis=1)
hatalar.columns = ["sales","predict_sales"]  #Sütun adı değiştirme
hatalar["hata"] = hatalar["sales"] - hatalar["predict_sales"]  #Yeni sütun ekleme
hatalar["hata_kareler"] = hatalar["hata"] ** 2
print("birim başına hata :",np.mean(hatalar["hata_kareler"]))



