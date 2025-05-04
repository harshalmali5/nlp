# natural_language_processing_demo.py

import numpy as np
import pandas as pd
import nltk

# Uncomment the following line if running this code for the first time
# nltk.download('all')  # Downloads all necessary NLTK data like tokenizers and corpora

# Sample sentence to work with
sentence = "If you want to know what a man’s like, take a good looks at how he treats his inferiors, not his equals. It is our choices, Harry, that show what we truly are, far more than our abilities!"

# 1. WhiteSpace Tokenization
from nltk.tokenize import WhitespaceTokenizer
whitespace_tokenized = WhitespaceTokenizer().tokenize(sentence)
print("1. Whitespace Tokenization:\n", whitespace_tokenized, '\n')

# 2. TreeBankWord Tokenization
from nltk.tokenize import TreebankWordTokenizer
treebank_tokenized = TreebankWordTokenizer().tokenize(sentence)
print("2. Treebank Tokenization:\n", treebank_tokenized, '\n')

# 3. MWE Tokenization (Multi-Word Expression Tokenizer)
from nltk.tokenize import MWETokenizer
# This tokenizer detects multi-word expressions.
mwe_tokenizer = MWETokenizer([('take', 'a', 'good')])
mwe_tokenizer.add_mwe(('Harry',))
mwe_tokenized = mwe_tokenizer.tokenize(list(sentence))  # MWE requires a list of tokens, here used per character
print("3. MWE Tokenization (Note: Character-wise input):\n", mwe_tokenized, '\n')

# 4. Tweet Tokenization
from nltk.tokenize import TweetTokenizer
tweet_tokenized = TweetTokenizer().tokenize(sentence)
print("4. Tweet Tokenization:\n", tweet_tokenized, '\n')

# 5. WordPunct Tokenization
from nltk.tokenize import wordpunct_tokenize
wordpunct_tokenized = wordpunct_tokenize(sentence)
print("5. WordPunct Tokenization:\n", wordpunct_tokenized, '\n')

# 6. Sentence Tokenization
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
sent_tokenized = sent_detector.tokenize(sentence)
print("6. Sentence Tokenization:\n", sent_tokenized, '\n')

# --------------------
# Stemming Techniques
# --------------------

# 1. Snowball Stemming
from nltk.stem.snowball import SnowballStemmer
snowballstemmer = SnowballStemmer(language='english')
snowball_stems = [snowballstemmer.stem(w) for w in tweet_tokenized]
print("Snowball Stemming:\nBefore:", tweet_tokenized, "\nAfter:", snowball_stems, '\n')

# 2. Porter Stemming
from nltk.stem import PorterStemmer
porterstemmer = PorterStemmer()
porter_stems = [porterstemmer.stem(w) for w in tweet_tokenized]
print("Porter Stemming:\nBefore:", tweet_tokenized, "\nAfter:", porter_stems, '\n')

# --------------------
# Lemmatization
# --------------------
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(w) for w in tweet_tokenized]
print("Lemmatization:\nBefore:", tweet_tokenized, "\nAfter:", lemmatized_words, '\n')
