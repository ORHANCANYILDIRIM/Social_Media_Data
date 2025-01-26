import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def predict_future(data, model, target_year):
    """
    Verilen hedef yıl için sosyal medya değişim oranlarını kullanarak intihar oranlarını tahmin eder.
    Hem geçmiş verileri hem de tahmin edilen verileri kullanarak geleceği tahmin eder.
    """

    # Geçmiş yıl verilerini al
    years = data["year"].values.reshape(-1, 1)
    twitter_change = data["Twitter user count % change since 2010"].values
    facebook_change = data["Facebook user count % change since 2010"].values
    instagram_change = data["Instagram user count % change since 2010 "].values

    # Her sosyal medya platformu için regresyon modelleri oluştur
    twitter_model = LinearRegression()
    facebook_model = LinearRegression()
    instagram_model = LinearRegression()

    # Geçmiş verilerle modelleri eğit
    twitter_model.fit(years, twitter_change)
    facebook_model.fit(years, facebook_change)
    instagram_model.fit(years, instagram_change)

    all_predictions = []
    social_media_data = []  # Sosyal medya verilerini saklayacağız
    future_years = np.arange(2020, target_year + 1).reshape(-1, 1)

    # Her yıl için adım adım tahmin yap
    for i in range(1, len(future_years) + 1):
        # Gelecek yıllar için tahmin yap (yıl 2020, 2021, 2022, vs.)
        years_for_prediction = future_years[:i]
        twitter_pred = twitter_model.predict(years_for_prediction)
        facebook_pred = facebook_model.predict(years_for_prediction)
        instagram_pred = instagram_model.predict(years_for_prediction)

        # Gelecek sosyal medya değişim oranlarını bir DataFrame'e ekleyelim
        future_socio_media_data = {
            'Year': future_years[:i].flatten(),
            'Twitter user count % change since 2010': twitter_pred,
            'Facebook user count % change since 2010': facebook_pred,
            'Instagram user count % change since 2010 ': instagram_pred
        }
        future_socio_media_data = pd.DataFrame(future_socio_media_data)

        # Sosyal medya verilerini sakla
        social_media_data.append(future_socio_media_data)

        # Modeli kullanarak intihar oranlarını tahmin et
        suicide_pred = model.predict(future_socio_media_data[['Twitter user count % change since 2010',
                                                              'Facebook user count % change since 2010',
                                                              'Instagram user count % change since 2010 ']])

        # Son tahmini alıyoruz
        all_predictions.append(suicide_pred[-1])

    # Hedef yıl için tahmin edilen intihar oranlarını ve sosyal medya verilerini döndür
    # Burada veri eksikse kontrol ekliyoruz
    twitter_change_vals = [item['Twitter user count % change since 2010'].iloc[-1] if not item[
        'Twitter user count % change since 2010'].empty else None for item in social_media_data]
    facebook_change_vals = [item['Facebook user count % change since 2010'].iloc[-1] if not item[
        'Facebook user count % change since 2010'].empty else None for item in social_media_data]
    instagram_change_vals = [item['Instagram user count % change since 2010 '].iloc[-1] if not item[
        'Instagram user count % change since 2010 '].empty else None for item in social_media_data]

    final_result = pd.DataFrame({
        'Year': future_years.flatten(),
        'Twitter Change': twitter_change_vals,
        'Facebook Change': facebook_change_vals,
        'Instagram Change ': instagram_change_vals,
        'Suicide Rate Prediction': all_predictions
    })

    return final_result



