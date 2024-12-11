import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import numpy as np

# Veri Yükleme
data = pd.read_excel("DataSet//social-media1.xlsx")

# Veriyi inceleyelim
print(data.info())

# Gereksiz sütunu kaldırma (Column7 tamamen NaN)
data = data.drop(columns=["Column7"])

# Eksik verileri kontrol etme
print("Eksik Veriler:")
print(data.isnull().sum())

# Veriyi görselleştirelim
# Yıl ve İntihar Oranı Değişimi
plt.figure(figsize=(10, 6))
plt.plot(data["year"], data["Suicide Rate % change since 2010"], marker='o')
plt.title("Yıllara Göre İntihar Oranı Değişimi", fontsize=14)
plt.xlabel("Yıl")
plt.ylabel("İntihar Oranı % Değişim")
plt.grid()
plt.show()

print(data.columns)

# Sosyal Medya Kullanımı ve İntihar Oranı
plt.figure(figsize=(10, 6))
plt.scatter(data["Twitter user count % change since 2010"], data["Suicide Rate % change since 2010"], alpha=0.7, label="Twitter")
plt.scatter(data["Facebook user count % change since 2010"], data["Suicide Rate % change since 2010"], alpha=0.7, label="Facebook")
plt.scatter(data["Instagram user count % change since 2010 "], data["Suicide Rate % change since 2010"], alpha=0.7, label="Instagram")
plt.title("Sosyal Medya Kullanımı ve İntihar Oranı", fontsize=14)
plt.xlabel("Sosyal Medya Kullanımı % Değişim")
plt.ylabel("İntihar Oranı % Değişim")
plt.legend()
plt.grid()
plt.show()

# Bağımsız değişkenler (Sosyal medya platformları kullanıcı sayısı değişimi) ve bağımlı değişken (İntihar oranı değişimi)
X = data[["Twitter user count % change since 2010",
          "Facebook user count % change since 2010",
          "Instagram user count % change since 2010 "]]
y = data["Suicide Rate % change since 2010"]

# Eğitim ve test verilerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluşturma ve eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Test verisiyle tahmin yapma
y_pred = model.predict(X_test)

# Tahmin edilen değerleri yazdır
print("Gerçek Değerler: ", y_test.values)
print("Tahmin Edilen Değerler: ", y_pred)

# Gerçek ve tahmin edilen değerleri karşılaştıran grafik
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Gerçek Değerler", marker='o')
plt.plot(y_pred, label="Tahmin Edilen Değerler", marker='x')
plt.title("Gerçek ve Tahmin Edilen Değerler", fontsize=14)
plt.xlabel("Test Verisi Örnekleri")
plt.ylabel("İntihar Oranı Değişimi (%)")
plt.legend()
plt.grid()
plt.show()


# Model değerlendirme
print("R² Skoru:", r2_score(y_test, y_pred))
print("Ortalama Kare Hata:", mean_squared_error(y_test, y_pred))




# Sadece sayısal sütunları seç
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Korelasyon hesapla
correlation = numeric_data.corr()

# Korelasyon matrisini yazdır
print(correlation)

# Korelasyon analizi

plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Korelasyon Matrisi", fontsize=14)
plt.show()

######### Gelecek tahmini ##########
print("###GELECEK TAHMİNİ "
      ""
      ""
      ""
)

# Sosyal medya kullanıcı değişimlerini ve yıl bilgisini alalım
years = data["year"].values.reshape(-1, 1)
twitter_change = data["Twitter user count % change since 2010"].values
facebook_change = data["Facebook user count % change since 2010"].values
instagram_change = data["Instagram user count % change since 2010 "].values

# Her bir sosyal medya platformu için Lineer Regresyon modelini eğitelim
twitter_model = LinearRegression()
facebook_model = LinearRegression()
instagram_model = LinearRegression()

# Modelleri eğitelim
twitter_model.fit(years, twitter_change)
facebook_model.fit(years, facebook_change)
instagram_model.fit(years, instagram_change)

# 2020, 2021, 2022 için tahmin yapalım (örnek olarak)
future_years = np.array([2020, 2021, 2022]).reshape(-1, 1)

twitter_pred = twitter_model.predict(future_years)
facebook_pred = facebook_model.predict(future_years)
instagram_pred = instagram_model.predict(future_years)

# Gelecek yıllar için tahminleri yazdıralım
print("2020-2022 yılları için Twitter kullanıcı değişimi tahminleri:", twitter_pred)
print("2020-2022 yılları için Facebook kullanıcı değişimi tahminleri:", facebook_pred)
print("2020-2022 yılları için Instagram kullanıcı değişimi tahminleri:", instagram_pred)

# Gelecek yıllar için tahmin verisini DataFrame formatına sokuyoruz
future_socio_media_data = pd.DataFrame({
    'Twitter user count % change since 2010': twitter_pred,
    'Facebook user count % change since 2010': facebook_pred,
    'Instagram user count % change since 2010 ': instagram_pred
})

# Modeli kullanarak intihar oranı değişimini tahmin edelim
suicide_pred = model.predict(future_socio_media_data)

print("2020-2022 yılları için İntihar Oranı Değişimi tahminleri:", suicide_pred)