import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd

from Social_Media.train_model import train_model
from grafik import plot_suicide_rate_change, plot_normalized_social_media_vs_suicide, plot_actual_vs_predicted, plot_correlation_matrix
from tahmin import predict_future

# data = None
#
# X_test =None
# y_test = None
# y_pred= None
# model = None

def deneme():
    if y_test is not None and y_pred is not None:
        print("boş değil ")
def load_data():
    """Veri dosyasını yükler ve model eğitir."""
    global data
    global X_test, y_test, y_pred
    global model
    # Veri Yükleme
    data = pd.read_excel("DataSet//social-media1.xlsx")

    # Veriyi inceleyelim
    print(data.info())
    print(data.columns)

    # Model ve verileri al
    model, X_test, y_test, y_pred = train_model(data)



def show_graph(graph_type):
    print("girdi1")
    # global X_test, y_test, y_pred
    # global data
    """Grafik çizimlerini çağırır."""
    if data is None:
        messagebox.showwarning("Uyarı", "Önce veri yükleyin!")
        return
    print("girdi2")

    try:
        if graph_type == "suicide_rate":
            print("girdi3")
            plot_suicide_rate_change(data)
        elif graph_type == "social_media":
            plot_normalized_social_media_vs_suicide(data)
        elif graph_type == "correlation":
            plot_correlation_matrix(data)
        elif graph_type == "actual_vs_predicted":
            if y_test is not None and y_pred is not None:
                plot_actual_vs_predicted(y_test, y_pred)  # Parametreleri fonksiyona gönder
            else:
                messagebox.showwarning("Uyarı", "Tahmin verileri eksik!")
    except Exception as e:
        messagebox.showerror("Hata", f"Grafik çizilirken bir hata oluştu:\n{e}")


def predict_future_years():
    global target_year

    target_year = int(year_entry.get())  # Kullanıcıdan alınan yıl
    if target_year < 2020 or target_year > 2029:
        messagebox.showerror("Hata", "Yıl 2020 ile 2029 arasında olmalıdır!")
        return

    result = predict_future(data, model, target_year)

    # Tahmin sonuçlarını formatla
    result_text = "Tahmin Sonuçları:\n\n"

    for index, row in result.iterrows():
        result_text += f"Yıl: {row['Year']}\n"
        result_text += f"Twitter Değişimi: {row['Twitter Change']:.2f}%\n"
        result_text += f"Facebook Değişimi: {row['Facebook Change']:.2f}%\n"
        result_text += f"Instagram Değişimi: {row['Instagram Change ']:.2f}%\n"
        result_text += f"İntihar Oranı Tahmini: {row['Suicide Rate Prediction']:.2f}%\n\n"

    messagebox.showinfo("Tahmin Sonuçları", result_text)


# Ana Pencere
root = tk.Tk()
root.title("Sosyal Medya ve İntihar Oranı Analizi")
root.geometry("600x500")

# Butonlar
load_data_button = tk.Button(root, text="Veri Yükle", command=load_data, width=20)
load_data_button.pack(pady=10)

suicide_rate_graph_button = tk.Button(root, text="İntihar Oranı Grafiği", command=lambda: show_graph("suicide_rate"), width=20)
suicide_rate_graph_button.pack(pady=10)

social_media_graph_button = tk.Button(root, text="Sosyal Medya Grafiği", command=lambda: show_graph("social_media"), width=20)
social_media_graph_button.pack(pady=10)

correlation_graph_button = tk.Button(root, text="Korelasyon Grafiği", command=lambda: show_graph("correlation"), width=20)
correlation_graph_button.pack(pady=10)

actual_vs_predicted_button = tk.Button(root, text="actual vs predicted Grafiği", command=lambda: show_graph("actual_vs_predicted"), width=20)
actual_vs_predicted_button.pack(pady=10)

# Kullanıcının hedef yılı girebileceği entry kutusu
year_label = tk.Label(root, text="Hedef Yılı Girin (2020-2029):")
year_label.pack(pady=5)

year_entry = tk.Entry(root)
year_entry.pack(pady=5)

# Gelecek tahminini başlatacak buton
predict_button = tk.Button(root, text="Gelecek Tahmini", command=predict_future_years, width=20)
predict_button.pack(pady=10)

# Çıkış Butonu
exit_button = tk.Button(root, text="Çıkış", command=root.quit, width=20, bg="red", fg="white")
exit_button.pack(pady=20)

# GUI döngüsü
root.mainloop()
