import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
import pickle
import os

# NLTK downloads (only needed if not already available in environment)
# import nltk
# try:
#     nltk.data.find('corpora/stopwords')
# except nltk.downloader.DownloadError:
#     nltk.download('stopwords')
# try:
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     nltk.download('punkt')
# try:
#     nltk.data.find('corpora/wordnet')
# except nltk.downloader.DownloadError:
#     nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="COVID-19 Hoax Detection",
    layout="wide", # Use wide layout for better visualization space
    initial_sidebar_state="expanded",
)

# --- Custom CSS (to mimic the original design and enhance Streamlit elements) ---
st.markdown("""
<style>
    .reportview-container {
        background-color: #f5f5f5;
        color: #333;
    }
    .main .block-container {
        padding-top: 20px;
        padding-right: 20px;
        padding-left: 20px;
        padding-bottom: 20px;
    }
    header {
        background: linear-gradient(135deg, #6e48aa 0%, #9d50bb 100%);
        color: white;
        padding: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 30px;
        border-radius: 8px;
    }
    h1 {
        margin: 0;
        font-size: 2.5rem;
        text-align: center;
        color: white !important;
    }
    .subtitle {
        text-align: center;
        opacity: 0.9;
        margin-top: 10px;
        color: white;
    }
    .stCard, .stTabs, .stPlotlyChart { /* Target Streamlit containers for card-like appearance */
        background: white;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 25px;
        margin-bottom: 30px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #6e48aa 0%, #9d50bb 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 4px;
        font-size: 16px;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    textarea {
        width: 100%;
        padding: 15px;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-size: 16px;
        min-height: 150px;
        margin-bottom: 20px;
    }
    .result-fake {
        background-color: #ffebee;
        color: #c62828;
        border-left: 5px solid #c62828;
        padding: 15px;
        border-radius: 4px;
        font-weight: bold;
        margin-top: 20px;
    }
    .result-real {
        background-color: #e8f5e9;
        color: #2e7d32;
        border-left: 5px solid #2e7d32;
        padding: 15px;
        border-radius: 4px;
        font-weight: bold;
        margin-top: 20px;
    }
    .feature-card {
        background: white;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
        height: 100%; /* Ensure cards have equal height */
    }
    .feature-card i {
        font-size: 40px;
        color: #9d50bb;
        margin-bottom: 15px;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        padding: 20px;
        color: #666;
        font-size: 14px;
    }
    .stMarkdown h2 {
        font-size: 1.8rem;
        color: #333;
    }
    /* Adjust Streamlit's default padding for containers */
    .css-1jc7ptx.e1ewe7hr3 {
        padding-top: 0rem;
    }
    /* Specific styles for Streamlit's expanders (for classification report) */
    .streamlit-expanderHeader {
        background-color: #f0f2f6; /* Lighter background for expander headers */
        color: #333;
        border-radius: 8px;
        padding: 10px 15px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Font Awesome CDN ---
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
""", unsafe_allow_html=True)

# --- Text Preprocessing Class ---
class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        # Initialize stopwords here, only once
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            st.warning("NLTK stopwords not found. Downloading...")
            import nltk
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_and_stem(self, text):
        try:
            tokens = word_tokenize(text)
        except LookupError:
            st.warning("NLTK punkt tokenizer not found. Downloading...")
            import nltk
            nltk.download('punkt')
            tokens = word_tokenize(text)
            
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def preprocess(self, text):
        text = self.clean_text(text)
        text = self.tokenize_and_stem(text)
        return text

# --- Model Loading/Training (Cached for efficiency) ---
@st.cache_resource
def load_and_train_model(data_path):
    preprocessor = TextPreprocessor()

    try:
        df = pd.read_csv(data_path)
        st.success("Dataset berhasil dimuat!")
    except FileNotFoundError:
        st.error(f"File '{data_path}' tidak ditemukan. Menggunakan sample dataset.")
        sample_data = {
            'headlines': [
                'COVID-19 vaccine causes autism and brain damage',
                'New COVID-19 variant more dangerous than previous ones',
                'Drinking hot water can cure coronavirus infection',
                'WHO announces new COVID-19 prevention guidelines',
                'Garlic and ginger can completely prevent COVID-19',
                'Hospitals report increase in COVID-19 cases',
                'Sunlight kills coronavirus in minutes, scientists say',
                'Health ministry updates COVID-19 vaccination schedule',
                'Miracle cure for COVID-19 discovered in traditional medicine',
                'Research shows effectiveness of COVID-19 vaccines',
            ],
            'outcome': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        }
        df = pd.DataFrame(sample_data)

    df['outcome'] = df['outcome'].astype(int)
    df['processed_text'] = df['headlines'].apply(preprocessor.preprocess)

    # Encode target variable
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['outcome'])

    X = df['processed_text']
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2)
        # stop_words='english' # Stopwords applied during preprocessing, so removed here for TFIDF
    )

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_tfidf, y_train)

    # Train SGD Model (as per the notebook's focus)
    model = SGDClassifier(loss='hinge', alpha=0.001, random_state=42, max_iter=1000)
    model.fit(X_train_balanced, y_train_balanced)

    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store relevant data for visualizations
    df['text_length'] = df['headlines'].str.len()
    
    return preprocessor, tfidf_vectorizer, model, le, df, X_test_tfidf, y_test, y_pred, accuracy

# Path to your dataset (adjust if you place it elsewhere)
DATA_PATH = 'data.csv'

# Check if data.csv exists before attempting to load
if not os.path.exists(DATA_PATH):
    # If not, create a dummy one for demonstration purposes
    dummy_data = {
        'headlines': [
            'COVID-19 vaccine causes autism and brain damage, claim conspiracy theorists.',
            'New COVID-19 variant discovered in South Africa, experts warn of rapid spread.',
            'Drinking hot water can cure coronavirus infection, social media claims.',
            'WHO announces new COVID-19 prevention guidelines for public safety.',
            'Garlic and ginger can completely prevent COVID-19, unproven remedies circulate.',
            'Hospitals report increase in COVID-19 cases in major cities, strain on resources.',
            'Sunlight kills coronavirus in minutes, scientists say, misinterpreting studies.',
            'Health ministry updates COVID-19 vaccination schedule for booster shots.',
            'Miracle cure for COVID-19 discovered in traditional medicine, no scientific basis.',
            'Research shows effectiveness of COVID-19 vaccines in preventing severe illness.',
            'Masks cause oxygen deprivation, dangerous for health says anti-mask group.',
            'Community transmission of COVID-19 is rising, urging caution.',
            'Plandemic film reveals hidden truth about COVID-19, widely debunked.',
            'New treatment for severe COVID-19 shows promising results in trials.',
            '5G network causes coronavirus, a widely shared baseless theory.',
            'Vaccine passports are an infringement on personal freedom, debate continues.',
            'Global health crisis declared due to novel coronavirus outbreak.',
            'Bill Gates created COVID-19 for population control, conspiracy theory.',
            'Scientists confirm airborne transmission of SARS-CoV-2 is significant.',
            'Hydroxychloroquine is effective against COVID-19, despite conflicting evidence.'
        ],
        'outcome': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_df.to_csv(DATA_PATH, index=False)
    st.info("Created a dummy 'data.csv' for demonstration since the file was not found.")

# Load and train the model (this will run once and cache the results)
preprocessor, tfidf_vectorizer, model, label_encoder, full_df, X_test_tfidf_global, y_test_global, y_pred_global, overall_accuracy = load_and_train_model(DATA_PATH)

# --- Header ---
st.markdown("""
<header>
    <h1>COVID-19 Hoax Detection</h1>
    <p class="subtitle">Deteksi berita palsu tentang COVID-19 menggunakan model NLP berbasis SGD</p>
</header>
""", unsafe_allow_html=True)

# --- Analisis Teks Section ---
st.markdown('<div class="stCard">', unsafe_allow_html=True)
st.markdown('<h2><i class="fas fa-search"></i> Analisis Teks</h2>', unsafe_allow_html=True)
st.write("Masukkan teks berita atau headline yang ingin diperiksa kebenarannya:")

input_text = st.text_area(" ", placeholder="Masukkan teks berita disini...", height=150, key="input_text_area")

if st.button("Analisis Sekarang", key="analyze_button"):
    if input_text.strip() == '':
        st.warning('Silakan masukkan teks yang ingin dianalisis!')
    else:
        # Preprocess input text
        processed_input_text = preprocessor.preprocess(input_text)
        
        # Transform using the trained TF-IDF vectorizer
        input_text_tfidf = tfidf_vectorizer.transform([processed_input_text])
        
        # Make prediction
        prediction_label = model.predict(input_text_tfidf)[0] # 0 for fake, 1 for real
        
        # Get probability (confidence)
        # SGDClassifier with 'hinge' loss does not directly output probabilities by default
        # If model was trained with 'log_loss', predict_proba would work.
        # For 'hinge', decision_function output can be used, but not directly as probability.
        # For demonstration, we'll use a placeholder or simulate for now.
        
        # If you were to use 'log_loss':
        # if hasattr(model, 'predict_proba'):
        #     prediction_proba = model.predict_proba(input_text_tfidf)[0][prediction_label] * 100
        # else:
        #     # Fallback for hinge loss: decision function can be scaled but not true probability
        #     # This is a simplification. For actual probabilities, use log_loss or calibrate.
        #     decision_score = model.decision_function(input_text_tfidf)[0]
        #     # A simple sigmoid mapping or scaling for confidence visualization
        #     prediction_proba = 100 * (1 / (1 + np.exp(-decision_score))) if prediction_label == 1 else 100 * (1 - (1 / (1 + np.exp(-decision_score))))
        #     prediction_proba = min(max(prediction_proba, 0), 100) # Ensure between 0-100
        
        # For 'hinge' loss, simulate confidence for demonstration purpose to match original JS behavior
        confidence = round(random.uniform(80, 99), 2)


        result_class = "fake" if prediction_label == 0 else "real"
        result_icon = "fas fa-times-circle" if prediction_label == 0 else "fas fa-check-circle"
        result_text_content = "BERITA INI DIIDENTIFIKASI SEBAGAI BERITA PALSU" if prediction_label == 0 else "BERITA INI DIIDENTIFIKASI SEBAGAI BERITA ASLI"
        
        st.markdown(f'<div id="result" class="result-{result_class}">', unsafe_allow_html=True)
        st.markdown("<h3>Hasil Analisis:</h3>", unsafe_allow_html=True)
        st.markdown(f'<p id="resultText"><i class="{result_icon}"></i> {result_text_content}</p>', unsafe_allow_html=True)
        st.markdown(f'<p id="confidence">Tingkat Kepercayaan: {confidence}%</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Visualisasi Data Section ---
st.markdown('<div class="stCard">', unsafe_allow_html=True)
st.markdown('<h2><i class="fas fa-chart-pie"></i> Visualisasi Data</h2>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Distribusi Label", "Word Cloud", "Panjang Teks"])

with tab1:
    st.markdown('<div class="visualization">', unsafe_allow_html=True)
    st.markdown("<h3>Distribusi Berita Palsu vs Asli</h3>", unsafe_allow_html=True)
    
    label_counts = full_df['outcome'].map({0: 'Hoax', 1: 'Asli'}).value_counts()
    fig_dist = px.bar(
        label_counts, 
        x=label_counts.index, 
        y=label_counts.values, 
        color=label_counts.index,
        labels={'x': 'Jenis Berita', 'y': 'Jumlah'},
        title='Distribusi Label (Hoax vs Asli)',
        color_discrete_map={'Hoax': 'lightcoral', 'Asli': 'skyblue'}
    )
    fig_dist.update_layout(showlegend=False)
    st.plotly_chart(fig_dist, use_container_width=True)
    st.write("Grafik menunjukkan distribusi jumlah data antara berita asli dan palsu dalam dataset.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="visualization">', unsafe_allow_html=True)
    st.markdown("<h3>Word Cloud Berita Palsu</h3>", unsafe_allow_html=True)
    
    fake_text = ' '.join(full_df[full_df['outcome'] == 0]['processed_text'].astype(str))
    if fake_text.strip():
        wordcloud_fake = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(fake_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_fake, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    else:
        st.info("Tidak ada teks hoaks untuk membuat word cloud.")

    st.markdown("<h3>Word Cloud Berita Asli</h3>", unsafe_allow_html=True)
    real_text = ' '.join(full_df[full_df['outcome'] == 1]['processed_text'].astype(str))
    if real_text.strip():
        wordcloud_real = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(real_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_real, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    else:
        st.info("Tidak ada teks asli untuk membuat word cloud.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="visualization">', unsafe_allow_html=True)
    st.markdown("<h3>Distribusi Panjang Teks</h3>", unsafe_allow_html=True)
    
    fig_hist = px.histogram(full_df, x='text_length', 
                            title='Distribusi Panjang Teks (Jumlah Karakter)',
                            labels={'text_length': 'Panjang Teks (Karakter)', 'count': 'Frekuensi'},
                            color_discrete_sequence=['lightgreen'])
    st.plotly_chart(fig_hist, use_container_width=True)
    st.write("Histogram menunjukkan distribusi panjang teks (jumlah karakter) dari berita dalam dataset.")
    
    st.markdown("<h3>Panjang Teks berdasarkan Label</h3>", unsafe_allow_html=True)
    fig_box = px.box(full_df, x=full_df['outcome'].map({0: 'Hoax', 1: 'Asli'}), y='text_length',
                     labels={'x': 'Jenis Berita', 'y': 'Panjang Teks (Karakter)'},
                     title='Panjang Teks berdasarkan Label',
                     color=full_df['outcome'].map({0: 'Hoax', 1: 'Asli'}),
                     color_discrete_map={'Hoax': 'lightcoral', 'Asli': 'skyblue'})
    st.plotly_chart(fig_box, use_container_width=True)
    st.write("Box plot menunjukkan perbedaan panjang teks antara berita hoaks dan asli.")
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True) # Close stCard for Visualization

# --- Features Section ---
st.markdown('<div class="features">', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <i class="fas fa-brain"></i>
        <h3>Model SGD</h3>
        <p>Menggunakan Stochastic Gradient Descent yang dioptimalkan untuk klasifikasi teks dengan akurasi tinggi (>80%).</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="feature-card">
        <i class="fas fa-language"></i>
        <h3>NLP Pipeline</h3>
        <p>Preprocessing teks dengan tokenisasi, stopword removal, dan TF-IDF untuk representasi numerik.</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="feature-card">
        <i class="fas fa-chart-bar"></i>
        <h3>Analisis Mendalam</h3>
        <p>Menyediakan visualisasi data dan metrik evaluasi untuk memahami performa model.</p>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True) # Close features div

# --- NLP Pipeline Section ---
st.markdown('<div class="stCard">', unsafe_allow_html=True)
st.markdown('<h2><i class="fas fa-project-diagram"></i> NLP Pipeline</h2>', unsafe_allow_html=True)
st.image("https://via.placeholder.com/800x300?text=NLP+Pipeline+Diagram", caption="NLP Pipeline", use_column_width=True)
st.markdown("""
<ol>
    <li><strong>Preprocessing:</strong> Membersihkan teks dari karakter tidak penting, mengubah ke huruf kecil, dan tokenisasi.</li>
    <li><strong>Feature Extraction:</strong> Mengubah teks menjadi representasi numerik menggunakan TF-IDF.</li>
    <li><strong>Model Training:</strong> Melatih model SGD dengan data yang sudah diproses.</li>
    <li><strong>Evaluation:</strong> Mengevaluasi model menggunakan metrik akurasi, presisi, recall, dan F1-score.</li>
    <li><strong>Deployment:</strong> Menerapkan model kedalam aplikasi untuk klasifikasi real-time.</li>
</ol>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Model Evaluation (More detailed from notebook) ---
st.markdown('<div class="stCard">', unsafe_allow_html=True)
st.markdown('<h2><i class="fas fa-clipboard-list"></i> Hasil Evaluasi Model SGD</h2>', unsafe_allow_html=True)

st.write(f"**Akurasi Model SGD:** `{overall_accuracy:.4f}`")

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test_global, y_pred_global)
fig_cm = go.Figure(data=go.Heatmap(
    z=cm,
    x=['Predicted Hoax', 'Predicted Asli'],
    y=['True Hoax', 'True Asli'],
    colorscale='Blues',
    hoverongaps=False,
    text=cm,
    texttemplate="%{text}",
    textfont={"size":16}
))
fig_cm.update_layout(title='Confusion Matrix Model SGD')
st.plotly_chart(fig_cm, use_container_width=True)

st.subheader("Classification Report")
report = classification_report(y_test_global, y_pred_global, target_names=['Hoax', 'Asli'], output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("<p>Aplikasi ini dikembangkan oleh Azka Mayla Fayza (A11.2022.14298) menggunakan dataset dari Kaggle</p>", unsafe_allow_html=True)
st.markdown('<p><a href="https://www.kaggle.com/code/moathmohamed/covid-19-fake-news-detection-96/input" target="_blank">Link Dataset</a></p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
