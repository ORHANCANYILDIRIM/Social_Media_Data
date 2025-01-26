from tkinter import messagebox

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def train_model(data):

        # Gereksiz sütunu kaldırma (sadece varsa)
        if "Column7" in data.columns:
            data = data.drop(columns=["Column7"])

        # Eksik verileri kontrol etme
        print("Eksik Veriler:")
        print(data.isnull().sum())

        # Bağımsız değişkenler ve bağımlı değişken
        try:
            X = data[["Twitter user count % change since 2010",
                      "Facebook user count % change since 2010",
                      "Instagram user count % change since 2010 "]]
            y = data["Suicide Rate % change since 2010"]
        except KeyError as e:
            print(f"Veri setinde beklenen sütun bulunamadı: {e}")
            return None, None, None

        # Eğitim ve test verilerine ayırma
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model oluşturma ve eğitme
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Test verisiyle tahmin yapma
        y_pred = model.predict(X_test)

        # Model değerlendirme
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        print("R² Skoru:", r2)
        print("Ortalama Kare Hata:", mse)

        # Messagebox ile sonucun gösterilmesi
        result_text = f"Model Değerlendirme Sonuçları:\n\n"
        result_text += f"R² Skoru: {r2:.4f}\n"
        result_text += f"Ortalama Kare Hata: {mse:.4f}\n"

        messagebox.showinfo("Model Değerlendirme", result_text)
        return model, X_test, y_test, y_pred