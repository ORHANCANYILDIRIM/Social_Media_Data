import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler



def plot_suicide_rate_change(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data["year"], data["Suicide Rate % change since 2010"], marker='o')
    plt.title("Yıllara Göre İntihar Oranı Değişimi", fontsize=14)
    plt.xlabel("Yıl")
    plt.ylabel("İntihar Oranı % Değişim")
    plt.grid()
    plt.show()





# Normalize edilmiş verilerle scatter plot
# Normalize edilmiş verilerle scatter plot
def plot_normalized_social_media_vs_suicide(data):
    # Verileri normalize etme
    scaler = MinMaxScaler()
    data_normalized = data.copy()

    # Normalizasyon uygulanacak sütunlar
    columns_to_normalize = [
        "Twitter user count % change since 2010",
        "Facebook user count % change since 2010",
        "Instagram user count % change since 2010 "
    ]

    # Belirtilen sütunları normalize et
    data_normalized[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

    # Normalize edilmiş verilerle grafiği çiz
    plt.figure(figsize=(10, 6))
    plt.scatter(
        data_normalized["Twitter user count % change since 2010"],
        data_normalized["Suicide Rate % change since 2010"],
        alpha=0.7, label="Twitter"
    )
    plt.scatter(
        data_normalized["Facebook user count % change since 2010"],
        data_normalized["Suicide Rate % change since 2010"],
        alpha=0.7, label="Facebook"
    )
    plt.scatter(
        data_normalized["Instagram user count % change since 2010 "],
        data_normalized["Suicide Rate % change since 2010"],
        alpha=0.7, label="Instagram"
    )
    plt.title("Normalize Edilmiş Sosyal Medya Kullanımı ve İntihar Oranı", fontsize=14)
    plt.xlabel("Sosyal Medya Kullanımı % Değişim (Normalize)")
    plt.ylabel("İntihar Oranı % Değişim")
    plt.legend()
    plt.grid()
    plt.show()




def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label="Gerçek Değerler", marker='o')
    plt.plot(y_pred, label="Tahmin Edilen Değerler", marker='x')
    plt.title("Gerçek ve Tahmin Edilen Değerler", fontsize=14)
    plt.xlabel("Test Verisi Örnekleri")
    plt.ylabel("İntihar Oranı Değişimi (%)")
    plt.legend()
    plt.grid()
    plt.show()



def plot_correlation_matrix(data):
    # Sayısal veriyi seç
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    correlation = numeric_data.corr()

    # Korelasyon matrisini çiz
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", cbar=True,
                annot_kws={"size": 10},  # Annotation font boyutunu ayarladık
                xticklabels=correlation.columns, yticklabels=correlation.columns)

    # Eksen etiketlerini döndürme
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(rotation=0, ha="right", fontsize=12)

    plt.title("Korelasyon Matrisi", fontsize=16)
    plt.tight_layout()  # Eksen etiketlerinin kesilmesini engeller
    plt.show()