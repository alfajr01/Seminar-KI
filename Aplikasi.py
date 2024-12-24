# Mengimpor pustaka yang diperlukan
import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
import pandas as pd
import numpy as np
import re
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
# Mengunduh daftar kata berhenti untuk bahasa Inggris dari NLTK
nltk.download('stopwords')
from sklearn.pipeline import Pipeline
import base64 


# Fungsi untuk membersihkan teks
def clean_text(text):
    # Menghapus tag HTML
    text = re.sub('<[^<]+?>', '', text)
    # Menghapus URL
    text = re.sub(r'http\S+', '', text)
    # Menghapus karakter non-alphanumeric
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Mengonversi ke huruf kecil
    text = text.lower()
    return text

# Membaca data dari Google Drive untuk Dataset Emotion dan Sentiment
data_emotion = pd.read_csv('Twitter_Emotion.csv')
data_sentiment = pd.read_csv('Data Train_Kartini_PP(550).csv')

# Fungsi untuk membersihkan teks
def clean_text(text):
    # Menghapus tag HTML
    text = re.sub('<[^<]+?>', '', text)
    # Menghapus URL
    text = re.sub(r'http\S+', '', text)
    # Menghapus karakter non-alphanumeric
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Mengonversi ke huruf kecil
    text = text.lower()
    return text

# Fungsi untuk membersihkan teks
def clean_text_new(text):
    # Menghapus tag HTML
    text = re.sub('<[^<]+?>', '', text)
    # Menghapus URL
    text = re.sub(r'http\S+', '', text)
    # Menghapus karakter non-alphanumeric
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Mengonversi ke huruf kecil
    text = text.lower()
    return text

# Fungsi untuk menghapus kata berhenti
def remove_stopwords(text):
    # Daftar kata berhenti
    stopwords = set(['dan', 'adalah', 'ini', 'pada', 'itu', 'di', 'dari', 'atau'])
    # Menghapus kata berhenti dari teks
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)

# Fungsi untuk melakukan lemmatisasi (contoh sederhana)
def lemmatize_text(text):
    lemmatized_words = [word.strip('.,') for word in text.split()]
    return ' '.join(lemmatized_words)

# Fungsi untuk memprediksi emosi dan sentimen dari teks
def predict_emotion_sentiment(text):
    # Membersihkan teks
    cleaned_text = clean_text_new(text)
    # Menghapus kata berhenti
    cleaned_text = remove_stopwords(cleaned_text)
    # Melakukan lemmatisasi
    cleaned_text = lemmatize_text(cleaned_text)
    # Melakukan prediksi menggunakan model Naive Bayes (contoh)
    y_pred_emotion = model_emotion.predict([cleaned_text])
    y_pred_sentiment = model_sentiment.predict([cleaned_text])
    return y_pred_emotion[0], y_pred_sentiment[0]


# Membersihkan teks pada kolom 'Description' dan 'review'
data_emotion['tweet'] = data_emotion['tweet'].apply(clean_text)
data_sentiment['Text Tweet'] = data_sentiment['Text Tweet'].apply(clean_text)

# Mengunduh daftar kata berhenti untuk bahasa Inggris dari NLTK
nltk.download('stopwords')

# Mengimpor daftar kata berhenti dari NLTK
from nltk.corpus import stopwords

# Mengambil daftar kata berhenti untuk bahasa Inggris
stop_words = set(stopwords.words('english'))

# Menghapus kata berhenti dari setiap kalimat dalam kolom 'Description' dan 'review'
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

data_emotion['tweet'] = data_emotion['tweet'].apply(remove_stopwords)
data_sentiment['Text Tweet'] = data_sentiment['Text Tweet'].apply(remove_stopwords)

# Inisialisasi tokenizer dan lemmatizer
w_tokenizer = WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer()

# Fungsi untuk melakukan lemmatisasi pada teks
def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])

# Terapkan lemmatisasi pada kolom 'Description' dan 'review'
data_emotion['tweet'] = data_emotion['tweet'].apply(lemmatize_text)
data_sentiment['Text Tweet'] = data_sentiment['Text Tweet'].apply(lemmatize_text)

# Mengambil nilai dari kolom 'Description' dan 'review' sebagai fitur
X_emotion = data_emotion['tweet'].values
X_sentiment = data_sentiment['Text Tweet'].values

# Mengambil nilai dari kolom 'emotion' dan 'sentiment' sebagai target
y_emotion = data_emotion['label'].values
y_sentiment = data_sentiment['Sentiment'].values

# Membagi data menjadi data pelatihan dan data uji
X_train_emotion, X_test_emotion, y_train_emotion, y_test_emotion = train_test_split(X_emotion, y_emotion, test_size=0.2, random_state=42)
X_train_sentiment, X_test_sentiment, y_train_sentiment, y_test_sentiment = train_test_split(X_sentiment, y_sentiment, test_size=0.2, random_state=42)

# Inisialisasi model Naive Bayes
model_emotion = make_pipeline(CountVectorizer(), MultinomialNB())
model_sentiment = make_pipeline(CountVectorizer(), MultinomialNB())

# Melatih model menggunakan data pelatihan
model_emotion.fit(X_train_emotion, y_train_emotion)
model_sentiment.fit(X_train_sentiment, y_train_sentiment)

# Melakukan prediksi pada data uji
y_pred_emotion = model_emotion.predict(X_test_emotion)
y_pred_sentiment = model_sentiment.predict(X_test_sentiment)

# Menghitung akurasi prediksi
accuracy_emotion = accuracy_score(y_test_emotion, y_pred_emotion)
accuracy_sentiment = accuracy_score(y_test_sentiment, y_pred_sentiment)

# Set page title
st.set_page_config(page_title="Analisis Sentimen Film", page_icon=":chart_with_upwards_trend:")

# Membuat sidebar selectbox
page = st.sidebar.selectbox("Go to", ["Beranda","Analisis File", "Analisis Teks"])

# Import library joblib
import joblib

# Save the emotion prediction model
joblib.dump(model_emotion, 'model_emotion.pkl')

# Save the sentiment prediction model
joblib.dump(model_sentiment, 'model_sentiment.pkl')

if page == "Beranda":
    st.title("Analisis Sentimen Film")
    st.write("Analisis sentimen adalah proses untuk mengidentifikasi dan mengklasifikasikan opini atau perasaan yang terkandung dalam teks, umumnya untuk mengetahui apakah suatu teks memiliki sentimen positif, negatif, atau netral. Analisis ini sering digunakan untuk memahami bagaimana orang merespon produk, layanan, merek, atau topik tertentu dalam media sosial, ulasan produk, komentar, artikel berita, dan berbagai sumber lainnya.")
    st.write("Dalam analisis sentimen, sistem komputer atau model machine learning digunakan untuk menilai bahasa alami dan mengategorikan teks berdasarkan emosi atau opini yang terkandung di dalamnya. Sentimen yang umum dianalisis meliputi:")
    st.write("1. Positif : Teks yang mengungkapkan perasaan senang, puas, atau dukungan.")
    st.write("2. Negatif : Teks yang mengungkapkan perasaan kecewa, marah, atau ketidakpuasan.")
    st.write("3. Netral  : Teks yang tidak menunjukkan emosi yang jelas atau hanya berfokus pada informasi tanpa opini kuat.")

elif page == "Analisis File":
    st.subheader("Akurasi Prediksi Pada Set Pengujian")
    accuracy_data = {
        "Task": ["Emotion", "Sentiment"],
        "Akurasi": [accuracy_emotion, accuracy_sentiment]
    }
    st.table(accuracy_data)

    st.title('Prediksi Emosi dan Sentimen Berdasarkan Inputan Dataset CSV')
    # Upload file
    uploaded_file = st.file_uploader("Upload dataset yang akan dilakukan prediksi emotion & sentiment", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            uji = pd.read_csv(uploaded_file, usecols=['full_text'])
        elif uploaded_file.name.endswith('.xlsx'):
            uji = pd.read_excel(uploaded_file, usecols=['full_text'])

        # Function to clean text
        def clean_text_new(text):
            # Remove HTML tags
            text = re.sub('<[^<]+?>', '', text)
            # Remove URLs
            text = re.sub(r'http\S+', '', text)
            # Remove non-alphanumeric characters
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            # Convert to lowercase
            text = text.lower()
            return text

        # Clean text in the new dataset
        uji['clean_text'] = uji['full_text'].apply(clean_text_new)

        # Remove stopwords from the cleaned text
        uji['clean_text'] = uji['clean_text'].apply(remove_stopwords)

        # Lemmatize the cleaned text
        uji['clean_text'] = uji['clean_text'].apply(lemmatize_text)

        # Get cleaned text as features for the new dataset
        X_uji = uji['clean_text'].values

        # Predict using the trained Naive Bayes models
        y_pred_uji_emotion = model_emotion.predict(X_uji)
        y_pred_uji_sentiment = model_sentiment.predict(X_uji)

        # Add prediction results to the new dataset DataFrame
        uji['predicted_emotion'] = y_pred_uji_emotion
        uji['predicted_sentiment'] = y_pred_uji_sentiment

        # Display the DataFrame
        st.dataframe(uji)
                # Download button for predicted dataset
        def download_link(df, filename):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download file</a>'
            return href

        #st.markdown(download_link(uji, 'predicted_results.csv'), unsafe_allow_html=True)
 
    else:
        st.write("Please upload a file to continue.")

elif page == "Analisis Teks":
    # Memuat model emosi dari file
    model_emotion = joblib.load('model_emotion.pkl')

    # Memuat model sentimen dari file
    model_sentiment = joblib.load('model_sentiment.pkl')

    # Judul aplikasi Streamlit
    st.title('Prediksi Emosi dan Sentimen Berdasarkan Inputan Teks')

    # Input teks untuk diuji
    text_to_test = st.text_input('Masukkan teks dalam bahasa Indonesia:')

    # Tombol untuk memulai prediksi
    if st.button('Prediksi'):
        # Melakukan prediksi emosi dan sentimen dari teks yang diberikan
        predicted_emotion, predicted_sentiment = predict_emotion_sentiment(text_to_test)

        # Menampilkan hasil prediksi
        st.subheader("Hasil Prediksi")
        accuracy_data = {
            "Task": ["Emotion", "Sentiment"],
            "Prediksi": [predicted_emotion, predicted_sentiment]
        }
        st.table(accuracy_data)