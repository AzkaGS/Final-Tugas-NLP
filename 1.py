# ===== INSTALL DEPENDENCIES =====
!pip install scikit-learn pandas numpy matplotlib seaborn wordcloud imblearn

# ===== IMPORT LIBRARIES =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Text processing
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# ===== LOAD DATASET =====
# Upload dataset terlebih dahulu ke Google Colab
# Uncomment line berikut jika ingin upload file
# from google.colab import files
# uploaded = files.upload()

# Load dataset (ganti dengan path file Anda)
try:
    df = pd.read_csv('/content/data.csv')
    print("Dataset berhasil dimuat!")
except FileNotFoundError:
    print("File tidak ditemukan. Silakan upload file 'data.csv' terlebih dahulu.")
    # Membuat sample data untuk demonstrasi
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
        'outcome': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] # Ensure outcome is int
    }
    df = pd.DataFrame(sample_data)
    print("Menggunakan sample dataset untuk demonstrasi.")

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Ensure 'outcome' is integer type
df['outcome'] = df['outcome'].astype(int)


# ===== EXPLORATORY DATA ANALYSIS (EDA) =====
print("\n===== EXPLORATORY DATA ANALYSIS =====")

# 1. Basic info about dataset
print("Dataset Info:")
print(df.info())
print("\nDataset Description:")
print(df.describe())

# 2. Check for missing values
print(f"\nMissing values:")
print(df.isnull().sum())

# 3. Check target distribution
print(f"\nTarget distribution:")
print(df['outcome'].value_counts())

# 4. Sample data
print(f"\nSample data:")
print(df.head())

# ===== DATA VISUALIZATION =====
print("\n===== DATA VISUALIZATION =====")

# Set style for plots
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Distribution of labels
df['outcome'].value_counts().plot(kind='bar', ax=axes[0,0], color=['skyblue', 'lightcoral'])
axes[0,0].set_title('Distribusi Label (Fake vs Real)')
axes[0,0].set_ylabel('Jumlah')
axes[0,0].set_xlabel('Label')
axes[0,0].tick_params(axis='x', rotation=45)

# 2. Text length distribution
df['text_length'] = df['headlines'].str.len()
df['text_length'].hist(bins=30, ax=axes[0,1], color='lightgreen', alpha=0.7)
axes[0,1].set_title('Distribusi Panjang Teks')
axes[0,1].set_ylabel('Frekuensi')
axes[0,1].set_xlabel('Panjang Teks')

# 3. Missing values heatmap
sns.heatmap(df.isnull(), cbar=True, ax=axes[1,0], cmap='viridis')
axes[1,0].set_title('Missing Values Heatmap')

# 4. Text length by label
df.boxplot(column='text_length', by='outcome', ax=axes[1,1])
axes[1,1].set_title('Panjang Teks berdasarkan Label')
axes[1,1].set_ylabel('Panjang Teks')

plt.tight_layout()
plt.show()

# ===== TEXT PREPROCESSING =====
print("\n===== TEXT PREPROCESSING =====")

class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)

        # Remove numbers
        text = re.sub(r'\d+', '', text)

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize_and_stem(self, text):
        """Tokenize and stem text"""
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        text = self.clean_text(text)
        text = self.tokenize_and_stem(text)
        return text

# Initialize preprocessor
preprocessor = TextPreprocessor()

# Apply preprocessing
print("Preprocessing text data...")
df['processed_text'] = df['headlines'].apply(preprocessor.preprocess)

# Show examples
print("\nContoh preprocessing:")
for i in range(3):
    print(f"Original: {df['headlines'].iloc[i]}")
    print(f"Processed: {df['processed_text'].iloc[i]}")
    print("-" * 50)

# ===== FEATURE ENGINEERING =====
print("\n===== FEATURE ENGINEERING =====")

# Add additional features
df['word_count'] = df['headlines'].str.split().str.len()
df['char_count'] = df['headlines'].str.len()
df['avg_word_length'] = df['char_count'] / df['word_count']

print("Additional features created:")
print(f"Word count range: {df['word_count'].min()} - {df['word_count'].max()}")
print(f"Character count range: {df['char_count'].min()} - {df['char_count'].max()}")

# ===== PREPARE DATA FOR MODELING =====
print("\n===== PREPARE DATA FOR MODELING =====")

# Encode target variable
le = LabelEncoder()
df['target'] = le.fit_transform(df['outcome'])

# Split data
X = df['processed_text']
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# ===== MODEL BUILDING =====
print("\n===== MODEL BUILDING =====")

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2),
    stop_words='english'
)

# Transform text data
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"TF-IDF shape: {X_train_tfidf.shape}")

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_tfidf, y_train)

print(f"Original training set distribution: {np.bincount(y_train)}")
print(f"Balanced training set distribution: {np.bincount(y_train_balanced)}")

# ===== MODEL TRAINING =====
print("\n===== MODEL TRAINING =====")

# Define models
models = {
    'SGD': SGDClassifier(loss='hinge', alpha=0.001, random_state=42, max_iter=1000),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(kernel='linear', random_state=42, probability=True),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate models
results = {}
predictions = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Fit model
    model.fit(X_train_balanced, y_train_balanced)

    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1] if hasattr(model, 'predict_proba') else None

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)

    # Store results
    results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

    predictions[name] = y_pred

    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ===== MODEL EVALUATION =====
print("\n===== MODEL EVALUATION =====")

# Create evaluation plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Model comparison
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]

axes[0,0].bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
axes[0,0].set_title('Perbandingan Akurasi Model')
axes[0,0].set_ylabel('Akurasi')
axes[0,0].set_ylim(0, 1)
for i, v in enumerate(accuracies):
    axes[0,0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

# 2. Confusion Matrix for best model
best_model = max(results, key=lambda x: results[x]['accuracy'])
cm = confusion_matrix(y_test, results[best_model]['predictions'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,1])
axes[0,1].set_title(f'Confusion Matrix - {best_model}')
axes[0,1].set_ylabel('True Label')
axes[0,1].set_xlabel('Predicted Label')

# 3. Cross-validation scores
cv_means = [results[name]['cv_mean'] for name in model_names]
cv_stds = [results[name]['cv_std'] for name in model_names]

axes[1,0].bar(model_names, cv_means, yerr=cv_stds, capsize=5,
              color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
axes[1,0].set_title('Cross-Validation Scores')
axes[1,0].set_ylabel('CV Score')
axes[1,0].set_ylim(0, 1)

# 4. Feature importance (for Random Forest)
if 'Random Forest' in models:
    rf_model = models['Random Forest']
    feature_names = tfidf.get_feature_names_out()
    importances = rf_model.feature_importances_

    # Get top 10 features
    top_indices = np.argsort(importances)[-10:]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = importances[top_indices]

    axes[1,1].barh(range(len(top_features)), top_importances)
    axes[1,1].set_yticks(range(len(top_features)))
    axes[1,1].set_yticklabels(top_features)
    axes[1,1].set_title('Top 10 Feature Importance (Random Forest)')
    axes[1,1].set_xlabel('Importance')

plt.tight_layout()
plt.show()

# ===== DETAILED CLASSIFICATION REPORTS =====
print("\n===== DETAILED CLASSIFICATION REPORTS =====")

for name, model in models.items():
    print(f"\n{name} Classification Report:")
    print("-" * 50)
    report = classification_report(y_test, results[name]['predictions'],
                                 target_names=['fake', 'real']) # Use explicit string labels
    print(report)

# ===== WORD CLOUDS =====
print("\n===== WORD CLOUDS =====")

# Create word clouds for fake and real news
fake_text = ' '.join(df[df['outcome'] == 0]['processed_text'].astype(str)) # Ensure text is string
real_text = ' '.join(df[df['outcome'] == 1]['processed_text'].astype(str)) # Ensure text is string

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Fake news word cloud
if fake_text.strip():
    wordcloud_fake = WordCloud(width=800, height=400, background_color='white').generate(fake_text)
    axes[0].imshow(wordcloud_fake, interpolation='bilinear')
    axes[0].set_title('Word Cloud - Fake News')
    axes[0].axis('off')

# Real news word cloud
if real_text.strip():
    wordcloud_real = WordCloud(width=800, height=400, background_color='white').generate(real_text)
    axes[1].imshow(wordcloud_real, interpolation='bilinear')
    axes[1].set_title('Word Cloud - Real News')
    axes[1].axis('off')

plt.tight_layout()
plt.show()

# ===== ERROR ANALYSIS =====
print("\n===== ERROR ANALYSIS =====")

# Analyze misclassified examples
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_predictions = results[best_model_name]['predictions']

# Find misclassified examples
misclassified_idx = np.where(y_test != best_predictions)[0]

print(f"Jumlah kesalahan klasifikasi: {len(misclassified_idx)}")
print(f"Persentase kesalahan: {len(misclassified_idx)/len(y_test)*100:.2f}%")

if len(misclassified_idx) > 0:
    print("\nContoh kesalahan klasifikasi:")
    test_indices = X_test.index[misclassified_idx[:5]]  # Show first 5 errors

    for idx in test_indices:
        true_label = le.inverse_transform([df.loc[idx, 'target']])[0]
        pred_label = le.inverse_transform([best_predictions[np.where(X_test.index == idx)[0][0]]])[0]

        print(f"\nText: {df.loc[idx, 'headlines']}")
        print(f"True Label: {true_label}")
        print(f"Predicted Label: {pred_label}")
        print("-" * 80)

# ===== SUMMARY AND RECOMMENDATIONS =====
print("\n===== SUMMARY AND RECOMMENDATIONS =====")

print("KESIMPULAN:")
print("=" * 50)
print(f"Model terbaik: {best_model_name}")
print(f"Akurasi terbaik: {results[best_model_name]['accuracy']:.4f}")
print(f"CV Score: {results[best_model_name]['cv_mean']:.4f}")

print("\nRanking Model berdasarkan Akurasi:")
sorted_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
for i, (name, result) in enumerate(sorted_models, 1):
    print(f"{i}. {name}: {result['accuracy']:.4f}")

print("\nREKOMENDASI:")
print("=" * 50)
print("1. Ensembling: Kombinasikan beberapa model terbaik untuk meningkatkan akurasi")
print("2. Data Augmentasi: Tambah data training dengan teknik augmentasi teks")
print("3. Hyperparameter Tuning: Optimasi parameter untuk model terbaik")
print("4. Deep Learning: Coba model BERT atau RoBERTa untuk hasil yang lebih baik")
print("5. Real-time Monitoring: Implementasi sistem monitoring untuk deployment")

# ===== SAVE RESULTS =====
print("\n===== SAVE RESULTS =====")

# Save model results to CSV
results_df = pd.DataFrame({
    'Model': model_names,
    'Accuracy': [results[name]['accuracy'] for name in model_names],
    'CV_Mean': [results[name]['cv_mean'] for name in model_names],
    'CV_Std': [results[name]['cv_std'] for name in model_names]
})

results_df.to_csv('model_results.csv', index=False)
print("Results saved to 'model_results.csv'")

# Save predictions
predictions_df = pd.DataFrame(predictions)
predictions_df['True_Label'] = y_test
predictions_df.to_csv('predictions.csv', index=False)
print("Predictions saved to 'predictions.csv'")

print("\n===== PROJECT COMPLETED =====")
print("Project NLP Klasifikasi Hoaks COVID-19 telah selesai!")
print("Semua visualisasi, analisis, dan hasil telah ditampilkan.")
print("File hasil telah disimpan untuk referensi lebih lanjut.")
