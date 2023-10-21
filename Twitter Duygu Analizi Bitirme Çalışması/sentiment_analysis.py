from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import pandas as pd

#duygu analizi
def sentiment_analysis(texts, model_name="dbmdz/bert-base-turkish-uncased"):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sa = pipeline("sentiment-analysis", tokenizer=tokenizer, model=model)

    labels, scores = [], []
    for text in texts:
        if isinstance(text, str):
            text_chunks = [text[i:i + 512] for i in range(0, len(text), 512)]
            text_results = []
            for chunk in text_chunks:
                result = sa(chunk)
                text_results.extend(result)

            label_0_score = next((r['score'] for r in text_results if r['label'] == 'LABEL_0'), 0.0)
            label_1_score = next((r['score'] for r in text_results if r['label'] == 'LABEL_1'), 0.0)

            if label_0_score > label_1_score:
                sentiment_label = "negatif"
                sentiment_score = label_0_score
            else:
                sentiment_label = "pozitif"
                sentiment_score = label_1_score

            labels.append(sentiment_label)
            scores.append(sentiment_score)
        else:
            labels.append(None)
            scores.append(None)

    return labels, scores

df = pd.read_csv('final.csv')
tweets = df['content'].tolist()

labels, scores = sentiment_analysis(tweets)

df['sentiment_label'] = labels
df['sentiment_score'] = scores

df.to_csv('duygu_analizi.csv', index=False)

duygu = pd.read_csv("duygu_analizi.csv")


duygu = duygu.drop(["Unnamed: 0", "date", "id"], axis=1)

duygu.to_csv("sentiment.csv")

duygu = pd.read_csv("sentiment.csv")
duygu.shape

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")

# Veri yükleme
duygu = pd.read_csv("sentiment.csv")

# Etiket adetlerini hesaplama
pozitif_adet = duygu[duygu["sentiment_label"] == "pozitif"].shape[0]
negatif_adet = duygu[duygu["sentiment_label"] == "negatif"].shape[0]

# Grafik renkleri
renk_pozitif = "#64B5F6"  # Mavi renk tonu
renk_negatif = "#FFB74D"  # Turuncu renk tonu

# Grafik oluşturma
labels = ["Pozitif", "Negatif"]
adetler = [pozitif_adet, negatif_adet]

fig, ax = plt.subplots()
bar_genislik = 0.9
bar1 = ax.bar(labels, adetler, bar_genislik, color=[renk_pozitif, renk_negatif])

# Değer etiketlerini ekleme
for rect in bar1:
    height = rect.get_height()
    ax.annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

# Eksen etiketleri ve başlıklarını ekleme
ax.set_xlabel('Duygu Etiketi')
ax.set_ylabel('Adet')
ax.set_title('Pozitif ve Negatif Yorumların Adedi')

# Arka plan rengini güncelleme
ax.set_facecolor("#F5F5F5")  # Açık gri arka plan rengi

# Çizgi ve çerçeve rengini güncelleme
ax.spines['bottom'].set_color("#CCCCCC")  # Alt çizgi rengi
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color("#CCCCCC")  # Sol çizgi rengi

# Grafik gösterme
plt.show()


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")

# Veri yükleme
duygu = pd.read_csv("sentiment.csv")

# Etiket adetlerini hesaplama
pozitif_adet = duygu[duygu["sentiment_label"] == "pozitif"].shape[0]
negatif_adet = duygu[duygu["sentiment_label"] == "negatif"].shape[0]

# Grafik renkleri
renk_pozitif = "#64B5F6"  # Mavi renk tonu
renk_negatif = "#FFB74D"  # Turuncu renk tonu

# Grafik oluşturma
labels = ["Pozitif", "Negatif"]
adetler = [pozitif_adet, negatif_adet]

fig, ax = plt.subplots()
bar_genislik = 0.9
bar1 = ax.bar(labels, adetler, bar_genislik, color=[renk_pozitif, renk_negatif])

# Değer etiketlerini ekleme
for rect in bar1:
    height = rect.get_height()
    ax.annotate('{}%'.format(round(height / sum(adetler) * 100, 2)),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

# Eksen etiketleri ve başlıklarını ekleme
ax.set_xlabel('Duygu Etiketi')
ax.set_ylabel('Adet')
ax.set_title('Pozitif ve Negatif Yorumların Adedi')

# Arka plan rengini güncelleme
ax.set_facecolor("#F5F5F5")  # Açık gri arka plan rengi

# Çizgi ve çerçeve rengini güncelleme
ax.spines['bottom'].set_color("#CCCCCC")  # Alt çizgi rengi
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color("#CCCCCC")  # Sol çizgi rengi

# Grafik gösterme
plt.show()

import matplotlib.pyplot as plt

# Veri yükleme
duygu = pd.read_csv("sentiment.csv")

# Etiket adetlerini hesaplama
pozitif_adet = duygu[duygu["sentiment_label"] == "pozitif"].shape[0]
negatif_adet = duygu[duygu["sentiment_label"] == "negatif"].shape[0]

# Grafik renkleri
renk_pozitif = "#64B5F6"  # Mavi renk tonu
renk_negatif = "#FFB74D"  # Turuncu renk tonu

# Grafik oluşturma
labels = ["Pozitif", "Negatif"]
adetler = [pozitif_adet, negatif_adet]

fig, ax = plt.subplots()
ax.pie(adetler, labels=labels, autopct='%1.1f%%', colors=[renk_pozitif, renk_negatif], startangle=90)
ax.set_title('Pozitif ve Negatif Yorumların Oranı')

# Grafik gösterme
plt.show()import matplotlib.pyplot as plt

# Veri yükleme
duygu = pd.read_csv("sentiment.csv")

# Etiket adetlerini hesaplama
pozitif_adet = duygu[duygu["sentiment_label"] == "pozitif"].shape[0]
negatif_adet = duygu[duygu["sentiment_label"] == "negatif"].shape[0]

# Grafik renkleri
renk_pozitif = "#64B5F6"  # Mavi renk tonu
renk_negatif = "#FFB74D"  # Turuncu renk tonu

# Grafik oluşturma
labels = ["Pozitif", "Negatif"]
adetler = [pozitif_adet, negatif_adet]

fig, ax = plt.subplots()
ax.pie(adetler, labels=labels, autopct='%1.1f%%', colors=[renk_pozitif, renk_negatif], startangle=90)
ax.set_title('Pozitif ve Negatif Yorumların Oranı')

# Grafik gösterme
plt.show()import matplotlib.pyplot as plt

# Veri yükleme
duygu = pd.read_csv("sentiment.csv")

# Etiket adetlerini hesaplama
pozitif_adet = duygu[duygu["sentiment_label"] == "pozitif"].shape[0]
negatif_adet = duygu[duygu["sentiment_label"] == "negatif"].shape[0]

# Grafik renkleri
renk_pozitif = "#64B5F6"  # Mavi renk tonu
renk_negatif = "#FFB74D"  # Turuncu renk tonu

# Grafik oluşturma
labels = ["Pozitif", "Negatif"]
adetler = [pozitif_adet, negatif_adet]

fig, ax = plt.subplots()
ax.pie(adetler, labels=labels, autopct='%1.1f%%', colors=[renk_pozitif, renk_negatif], startangle=90)
ax.set_title('Pozitif ve Negatif Yorumların Oranı')

# Grafik gösterme
plt.show()


import matplotlib.pyplot as plt

# Veri yükleme
duygu = pd.read_csv("sentiment.csv")

#------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Veri yükleme
duygu = pd.read_csv("sentiment.csv")

# Kategoriler
kategoriler = ["ekonomik", "sosyal", "siyasi", "diğer"]

# Pozitif ve negatif yorum adetlerini hesaplama
pozitif_adetler = []
negatif_adetler = []

for kategori in kategoriler:
    pozitif_adet = duygu[(duygu["sentiment_label"] == "pozitif") & (duygu["tahmin_decision_tree"] == kategori)].shape[0]
    negatif_adet = duygu[(duygu["sentiment_label"] == "negatif") & (duygu["tahmin_decision_tree"] == kategori)].shape[0]
    pozitif_adetler.append(pozitif_adet)
    negatif_adetler.append(negatif_adet)

# Grafik renkleri
renk_pozitif = "#4682b4"  # Mavi renk tonu
renk_negatif = "#FF4500"  # Turuncu renk tonu

# Grafik oluşturma
fig, ax = plt.subplots()
bar_genislik = 0.4
indeks = np.arange(len(kategoriler))

bar1 = ax.bar(indeks, pozitif_adetler, bar_genislik, label="Pozitif", color=renk_pozitif)
bar2 = ax.bar(indeks + bar_genislik, negatif_adetler, bar_genislik, label="Negatif", color=renk_negatif)

# Değer etiketlerini ekleme
for rect in bar1 + bar2:
    height = rect.get_height()
    ax.annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

# Eksen etiketleri ve başlıklarını ekleme
ax.set_xlabel('Tahmin_decision_tree')
ax.set_ylabel('Tweet Adedi')
ax.set_title('Pozitif ve Negatif Yorum Adetleri - Tahmin Decision Tree')
ax.legend()

# Arka plan rengini güncelleme
ax.set_facecolor("#F5F5F5")  # Açık gri arka plan rengi

# Çizgi ve çerçeve rengini güncelleme
ax.spines['bottom'].set_color("#CCCCCC")  # Alt çizgi rengi
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color("#CCCCCC")  # Sol çizgi rengi

# X ekseni düzenlemeleri
ax.set_xticks(indeks + bar_genislik / 2)
ax.set_xticklabels(kategoriler)

# Grafik gösterme
plt.show()


#-------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# Veri yükleme
duygu = pd.read_csv("sentiment.csv")

# Kategoriler
kategoriler = ["ekonomik", "sosyal", "siyasi", "diğer"]

# Pozitif ve negatif yorum adetlerini hesaplama
pozitif_adetler = []
negatif_adetler = []

for kategori in kategoriler:
    pozitif_adet = duygu[(duygu["sentiment_label"] == "pozitif") & (duygu["tahmin_gradient_boosting"] == kategori)].shape[0]
    negatif_adet = duygu[(duygu["sentiment_label"] == "negatif") & (duygu["tahmin_gradient_boosting"] == kategori)].shape[0]
    pozitif_adetler.append(pozitif_adet)
    negatif_adetler.append(negatif_adet)

# Grafik renkleri
renk_pozitif = "#4682b4"  # Mavi renk tonu
renk_negatif = "#FF4500"  # Turuncu renk tonu

# Grafik oluşturma
fig, ax = plt.subplots()
bar_genislik = 0.4
indeks = np.arange(len(kategoriler))

bar1 = ax.bar(indeks, pozitif_adetler, bar_genislik, label="Pozitif", color=renk_pozitif)
bar2 = ax.bar(indeks + bar_genislik, negatif_adetler, bar_genislik, label="Negatif", color=renk_negatif)

# Değer etiketlerini ekleme
for rect in bar1 + bar2:
    height = rect.get_height()
    ax.annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

# Eksen etiketleri ve başlıklarını ekleme
ax.set_xlabel('Tahmin_gradient_boosting')
ax.set_ylabel('Tweet Adedi')
ax.set_title('Pozitif ve Negatif Yorum Adetleri - Tahmin Gradient Boosting')
ax.legend()

# Arka plan rengini güncelleme
ax.set_facecolor("#F5F5F5")  # Açık gri arka plan rengi

# Çizgi ve çerçeve rengini güncelleme
ax.spines['bottom'].set_color("#CCCCCC")  # Alt çizgi rengi
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color("#CCCCCC")  # Sol çizgi rengi

# X ekseni düzenlemeleri
ax.set_xticks(indeks + bar_genislik / 2)
ax.set_xticklabels(kategoriler)

# Grafik gösterme
plt.show()