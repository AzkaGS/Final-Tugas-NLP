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

# --- NLTK Downloads (ENSURE THESE RUN FIRST AND SUCCESSFULLY) ---
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    st.info("Downloading NLTK stopwords...")
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    st.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

# The error explicitly mentions 'punkt_tab/english/'. While 'punkt' usually covers it,
# sometimes specific modules look for this exact path. Let's try downloading
# 'averaged_perceptron_tagger' which is often a dependency for tokenizers,
# and also 'wordnet' for completeness if you use it (your original notebook did).
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    st.info("Downloading NLTK averaged_perceptron_tagger...")
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    st.info("Downloading NLTK wordnet...")
    nltk.download('wordnet')


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
        background-color: #e8f5e
