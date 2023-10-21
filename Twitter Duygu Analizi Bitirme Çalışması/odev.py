#Aşağıdaki kod bloğu çalışmam için gerekli kütüphaneleri ve dosyaları, çalışmama dahil edebilmek için yazılmıştır.
from utils import *
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

dataframe = pd.read_csv('enflasyontr.csv')
dataframe["content"].head(10)


#------------------------------------------------------------------------------------------------------------------------------------------------------

#Aşağıdaki kod satırı tweet çekmek için yazılmıştır; query, limit, also_csv ve csv_name parametrelerinden oluşmaktadır.
#query: Twitter da arama yaptığımız sorguyu yazmak için kullanılan bir parametre.
#limit: Alınacak tweetlerin maksimum sayısını belirler.
#also_csv: True olarak kaydedildiğinde bir csv dosyası oluşturur.
#csv_name: Oluşturulan csv dosyalarına isim vermek için kullanılan bir parametredir.
tweets = get_tweets('"enflasyon" lang:tr until:2023-01-04 since:2006-01-01', limit=100001, also_csv=True, csv_name='enflasyontr.csv')

#Bu kod satırında tweetleri içerisinde barındıran csv dosyasını okutuyoruz.
dataframe = pd.read_csv('enflasyontr.csv')

#Aşağıdaki kod okuttuğumuz dosyanın içerisinde kaç satır ve kaç sütun olduğunu gösterir.
dataframe.shape

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Aşağıdaki kodlarla csv dosyasının içerisindeki istenmeyen sütunları kaldırdık. Axis parametresi 0 değerini aldığında satırlar, 1 değerini aldığında ise sütunlar üzerinde işlem yapılır.
dataframe = dataframe.drop(['url', 'media', 'hashtags', 'place', 'user', 'user_name', 'retweet_count', 'like_count',
                            'quoteCount', 'lang', 'user_location', 'cashtags', 'conversation_id', 'coordinates', 'inReplyToTweetId',
                            'inReplyToUser', 'mentionedUsers', 'out_links', 'quotedTweet', 'renderedContent', 'replyCount',
                            'retweetCount', 'retweetedTweet', 'source', 'sourceLabel', 'sourceUrl', 'tco_out_links', 'user_created',
                            'user_description', 'user_descriptionUrls', 'user_display_name', 'user_favouritesCount', 'user_followersCount',
                            'user_friendsCount', 'user_id', 'user_label', 'user_link_Tco_url', 'user_linkUrl', 'user_listedCount',
                            'user_location', 'user_media_count', 'user_profile_banner_url', 'user_profile_image_url', 'user_protected',
                            'user_raw_description', 'user_statuses_count', 'user_url', 'user_username', 'user_verified'], axis=1)

#Aşağıdaki kod satırında, teyit etmek amaçlı veri setinin sütunlarını döndürmesini istedim.
dataframe.columns

#Sütunlara baktığımda "user_location.1" diye bir sütun olduğunu gördüm ve aşağıdaki kod satırı ile onu da kaldırmak istedim.
dataframe = dataframe.drop('user_location.1', axis=1)

#----------------------------------------------------------------------------------------------------------------------------------------------------------

#Aşağıdaki kod satırı içeriğin küçük harf yapılması için kullanılmıştır.
dataframe["content"] = dataframe["content"].apply(lambda x: " ".join(x.lower() for x in x.split()))

#Aşağıdaki kod satırı hashtag'lerin kaldırılması için kullanılmıştır.
dataframe["content"] = dataframe["content"].str.replace('#[A-Za-z0-9]+\s?', '', regex=True)

#Aşağıdaki kod satırı mentions'ların kaldırılması için kullanılmıştır.
dataframe["content"] = dataframe["content"].str.replace('@[A-Za-z0-9]+\s?', '', regex=True)

#Yeni satırların kaldırılması
dataframe["content"] = dataframe["content"].str.replace(r'\n', '', regex=True)

#Sayıların kaldırılması
dataframe["content"] = dataframe["content"].str.replace(r'\d+', '', regex=True)

#Aşağıdaki kod satırı harf uzunluğu 2'den az olanların (Örneğin ve, ya, bu...) kaldırılması için kullanılmıştır.
dataframe["content"] = dataframe["content"].apply(lambda x: re.sub(r'\b\w{1,2}\b', '', x))

#Noktalama işaretlerinin kaldırılması
dataframe["content"] = dataframe["content"].str.replace(r'[^\w\s]', '', regex=True)

#Aşağıdaki kod satırı alt çizgileri kaldırmak için kullanılmıştır.
dataframe["content"] = dataframe["content"].str.replace("_", "")

#Aşağıdaki kod satırı linklerin kaldırılması için kullanılmıştır.
dataframe["content"] = dataframe["content"].str.replace(r'http\S+', '', regex=True)

#Aşağıdaki kod satırı stopwords("veya", "ama", "ise", "bir", "birkaç") kaldırılması için kullanılmıştır.
nltk.download("stopwords")
stop_w = stopwords.words("turkish")
dataframe["content"] = dataframe["content"].astype(str)
dataframe["content"] = dataframe["content"].apply(lambda x: " ".join(x for x in str(x).split() if x not in stop_w))

#Aşağıdaki kod temizlenmiş verilerden yeni bir veri seti oluşturmak için kullanılır.
dataframe.to_csv("enf_temiz_veri.csv")

#------------------------------------------------------------------------------------------------------------------------------------------------------

#Aşağıdaki kod yeni veri setini okutmak için kullanılmıştır.
dataframe = pd.read_csv('enf_temiz_veri.csv')
dataframe["content"].head()#Bu kod satırı yaptığımız veri temizleme işlemini teyit etmek için kullanılmıştır.

#Aşağıdaki kodlarla kategorilere ayırdığımız verileri tutan veri setini okutarak teyit etmek amaçlı ilk 10 veriyi döndürmesini istedim.
learning_set = pd.read_csv('category_enflasyon.csv', sep=",")
learning_set.head(10)
learning_set.shape

#------------------------------------------------------------------------------------------------------------------------------------------------------

#Aşağıdaki kodlar eğitim ve test veri kümelerini oluşturmak için kullanılmıştır.
df = pd.read_csv("enf_temiz_veri.csv")
train_df = pd.read_csv("category_enflasyon.csv", sep=",")
model_df = train_df[['text', 'category']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(model_df["text"], model_df["category"], test_size=0.2, random_state = 4)
X_train.shape
y_train.shape
X_test.shape
y_test.shape

#------------------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer #Import ettiğimiz TfidfVectorizer metin verilerini TF-IDF vektörlerine dönüştürmek için kullanılan bir sklearn kütüphanesi sınıfıdır.
#Aşağıdaki kod bloğu metinlerin makine öğrenimi modellerine girdi olarak verilebilmesi için metinleri TF-IDF vektörlerine dönüştürür.
tfidf_vectorizer = TfidfVectorizer(analyzer='word')
tfidf_learning_vec = tfidf_vectorizer.fit(learning_set.text)
tfidf_wm = tfidf_learning_vec.transform(learning_set.text)
tfidf_tokens = tfidf_vectorizer.get_feature_names_out()
df_tfidf_vect = pd.DataFrame(data=tfidf_wm.toarray(), columns=tfidf_tokens)
df_tfidf_vect

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)
print(train_vectors.shape, test_vectors.shape)
print(train_vectors)

#------------------------------------------------------------------------------------------------------------------------------------------------------

#Aşağıdaki kodlar makine öğrenmesi algoritmalarını kullanmak için yazılmıştır. Burada çeşitli sınıflandırma algoritmaları ve metrikler bulunmaktadır.
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBRFClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import seaborn as sns

#------------------------------------------------------------------------------------------------------------------------------------------------------

#Aşğıdaki kodlar modellerin performansını değerlendirmek için kullanılan algoritmalardır.
clf = MultinomialNB()
clf.fit(train_vectors, y_train)
prediction_ml = clf.predict(test_vectors)
print("Naive Bayes :\n", confusion_matrix(y_test, prediction_ml), "\n")
print(accuracy_score(y_test, prediction_ml))

LogicReg = LogisticRegression()
LogicReg.fit(train_vectors, y_train)
prediction_lr =  LogicReg.predict(test_vectors)
print("Logistic Regression :\n", confusion_matrix(y_test, prediction_lr), "\n")
print(accuracy_score(y_test, prediction_lr))

dTmodel = DecisionTreeClassifier()
dTmodel.fit(train_vectors, y_train)
prediction_dt = dTmodel.predict(test_vectors)
print("DecisionTree :\n", confusion_matrix(y_test, prediction_dt), "\n")
print(accuracy_score(y_test, prediction_dt))

rForest = RandomForestClassifier()
rForest.fit(train_vectors, y_train)
prediction_rf = rForest.predict(test_vectors)
print("RandomForest :\n", confusion_matrix(y_test, prediction_rf), "\n")
print(accuracy_score(y_test, prediction_rf))

grBoosting = GradientBoostingClassifier()
grBoosting.fit(train_vectors, y_train)
prediction_gb = grBoosting.predict(test_vectors)
print("GradientBoosting :\n", confusion_matrix(y_test, prediction_gb), "\n")
print(accuracy_score(y_test, prediction_gb))

df["content"]
test_vectors_ = vectorizer.transform(df["content"].astype('U').values)
print(test_vectors_.shape)
print(test_vectors_)

#------------------------------------------------------------------------------------------------------------------------------------------------------

#TAHMİNLER
#logistic
predicted = LogicReg.predict(test_vectors_)
tahmin = pd.DataFrame(predicted)
tahmin.rename(columns={0: 'tahmin'}, inplace=True)
df["tahmin_logistic"] = tahmin
df.head(20)


#naive bayes
predicted = clf.predict(test_vectors_)
tahmin = pd.DataFrame(predicted)
tahmin.rename(columns= {0:"tahmin"}, inplace=True)
df["tahmin_naive_bayes"] = tahmin
df.head(20)

#decision tree
predicted = clf.predict(test_vectors_)
tahmin = pd.DataFrame(predicted)
tahmin.rename(columns= {0:"tahmin"}, inplace=True)
df["tahmin_decision_tree"] = tahmin
df.head(20)

#random forest
predicted = clf.predict(test_vectors_)
tahmin = pd.DataFrame(predicted)
tahmin.rename(columns= {0:"tahmin"}, inplace=True)
df["tahmin_random_forest"] = tahmin
df.head(20)

#gradient boosting
predicted = clf.predict(test_vectors_)
tahmin = pd.DataFrame(predicted)
tahmin.rename(columns= {0:"tahmin"}, inplace=True)
df["tahmin_gradient_boosting"] = tahmin
df.head(20)

df.to_csv("final.csv")

#Verilerin görselleştirilmesi
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")

algoritmalar = ['Naive Bayes', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting']
accuracies = [accuracy_score(y_test, prediction_ml),
              accuracy_score(y_test, prediction_lr),
              accuracy_score(y_test, prediction_dt),
              accuracy_score(y_test, prediction_rf),
              accuracy_score(y_test, prediction_gb)]

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(algoritmalar, accuracies, color=['lightblue', 'lightgreen', 'lightpink', 'lightgray', 'lightsalmon'])

ax.set_xlabel('Başarı Oranı')
ax.set_ylabel('Algoritmalar')

for bar in bars:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.2%}', ha='left', va='center')

plt.show()




