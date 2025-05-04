# nlp_preprocessing.py

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

# Ensure necessary NLTK data is downloaded
# nltk.download('punkt')
# nltk.download('stopwords')

# Load dataset
data = pd.read_csv('../Data/assignment2.csv')

# Display initial info (optional)
# print(data.head())
# print(data.describe(include='all'))

# Drop rows with missing values and reset the index
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

# Check for any null values after cleanup (optional)
# print(data.isnull().sum())

# Preprocessing function to tokenize text, lowercase it, remove stopwords and non-alphabetic tokens
stop_words = set(stopwords.words('english'))
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return tokens

# Apply preprocessing to the 'Market Category' column
data['Market Category'] = data['Market Category'].apply(preprocess)

# ----------------- Bag-of-Words -----------------
count_vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
bow = count_vectorizer.fit_transform(data['Market Category'])

# Print first 5 rows and 10 columns of BoW matrix
print("Bag-of-Words Sample:")
print(bow.toarray()[:5, :10])

# ----------------- Normalized Count Occurrence -----------------
normalized_count = bow.copy()
for i, j in zip(*normalized_count.nonzero()):
    doc_length = len(data['Market Category'][i])
    if doc_length > 0:
        normalized_count[i, j] = normalized_count[i, j] / doc_length

print("\nNormalized Count Occurrence Sample:")
print(normalized_count.toarray()[:5, :10])

# ----------------- TF-IDF -----------------
tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False)
tfidf = tfidf_vectorizer.fit_transform(data['Market Category'])

print("\nTF-IDF Sample:")
print(tfidf.toarray()[:5, :10])

# ----------------- Word2Vec Embedding -----------------
# Train Word2Vec model on tokenized data
model = Word2Vec(sentences=data['Market Category'], vector_size=100, window=5, min_count=1, workers=4)

# Initialize empty embedding matrix
embeddings = np.zeros((len(data), 100))

# Average word vectors for each document
for i, tokens in enumerate(data['Market Category']):
    if tokens:  # Avoid division by zero
        for token in tokens:
            embeddings[i] += model.wv[token]
        embeddings[i] /= len(tokens)

print("\nWord2Vec Embeddings Sample:")
print(embeddings[:5])

