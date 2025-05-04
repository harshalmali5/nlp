# news_classification_pipeline.py

import pandas as pd
import re
import nltk
import pickle
import numpy as np
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2

# --- Download NLTK Resources ---
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# --- Load Dataset ---
with open('./Data/News_dataset.pickle', 'rb') as f:
    df = pd.read_pickle(f)

# --- Basic Cleaning ---
df['Content_Parsed_1'] = df['Content'].str.replace("\r", " ").str.replace("\n", " ").str.replace("    ", " ").str.replace('"', '')
df['Content_Parsed_2'] = df['Content_Parsed_1'].str.lower()

# --- Remove Punctuation ---
punctuation_signs = list("?:!.,;")
df['Content_Parsed_3'] = df['Content_Parsed_2']
for punct_sign in punctuation_signs:
    df['Content_Parsed_3'] = df['Content_Parsed_3'].str.replace(punct_sign, '', regex=True)

df['Content_Parsed_4'] = df['Content_Parsed_3'].str.replace("'s", "", regex=True)

# --- Lemmatization ---
lemmatizer = WordNetLemmatizer()
lemmatized_texts = []
for text in df['Content_Parsed_4']:
    tokens = text.split(" ")
    lemmatized = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
    lemmatized_texts.append(" ".join(lemmatized))

df['Content_Parsed_5'] = lemmatized_texts

# --- Stopword Removal ---
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stop_words])

df['Content_Parsed_6'] = df['Content_Parsed_5'].apply(remove_stopwords)

# --- Column Selection ---
df = df[["File_Name", "Category", "Complete_Filename", "Content", "Content_Parsed_6"]]
df = df.rename(columns={'Content_Parsed_6': 'Content_Parsed'})

# --- Label Encoding ---
category_codes = {'business': 0, 'entertainment': 1, 'politics': 2, 'sport': 3, 'tech': 4}
df['Category_Code'] = df['Category'].map(category_codes)

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    df['Content_Parsed'], df['Category_Code'], test_size=0.15, random_state=8
)

# --- TF-IDF Vectorization ---
tfidf = TfidfVectorizer(
    encoding='utf-8',
    ngram_range=(1, 2),
    stop_words=None,
    lowercase=False,
    max_df=1.0,
    min_df=10,
    max_features=300,
    norm='l2',
    sublinear_tf=True
)

features_train = tfidf.fit_transform(X_train).toarray()
features_test = tfidf.transform(X_test).toarray()

# --- Chi-Square Feature Selection ---
for label, category_id in category_codes.items():
    chi2score = chi2(features_train, y_train == category_id)
    indices = np.argsort(chi2score[0])
    feature_names = np.array(tfidf.get_feature_names_out())[indices]
    unigrams = [term for term in feature_names if len(term.split(' ')) == 1]
    bigrams = [term for term in feature_names if len(term.split(' ')) == 2]

    print(f"# '{label}' category:")
    print("  . Most correlated unigrams:\n. " + "\n. ".join(unigrams[-5:]))
    print("  . Most correlated bigrams:\n. " + "\n. ".join(bigrams[-2:]))
    print("")

# Optional: Save processed data or vectorizer
# with open("tfidf_vectorizer.pkl", "wb") as f:
#     pickle.dump(tfidf, f)
