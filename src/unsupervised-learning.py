import math
import pandas as pd
import numpy as np
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
#from fpdf import FPDF
import base64
from langdetect import detect
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

nltk.download('punkt')

# Load dataset
df = pd.read_csv('.static/sankhya-karika-sanskrit-to-english.csv', names=['Sanskrit-shloka', 'English-translation', 'English-meaning'])

# detect language
df['sanskrit-language'] = df['Sanskrit-shloka'].apply(str).apply(lambda x: detect(x))
df['english-language'] = df['English-translation'].apply(str).apply(lambda x: detect(x))

# tokenize
df["english-translation-tokens"] = df['English-translation'].replace(to_replace='-', value=' ', regex=True).apply(word_tokenize)

# remove punctuations
custom_punctuation_list = {'|', '||'}
def remove_punct(token):
    return [word for word in token if word.isalpha() and word not in custom_punctuation_list]

df["english-translation-tokens"] = df["english-translation-tokens"].apply(remove_punct)

# stemming: Stemming is the process of reducing the words to their word stem or root form.


# Lemmatization: Unlike stemming, lemmatization reduces words to their base word, reducing the inflected words properly and ensuring that the root word belongs to the language. 

ignore_word_list = {'ca', 'tat', 'api', 'eva', 'yathā', 'tathā', 'ye', 'te', 'bhavati', 'tu', 'na', 'hi', 'tad', 'iti', 'ete', 'asmāt', 'tasmāt'}
word_list = pd.Series([word for tokens in df["english-translation-tokens"] for word in tokens if word not in ignore_word_list])
# print(word_list)

word_counts = pd.Series(word_list).value_counts()
print(word_counts)

df.to_csv('.static/sankhya-karika-sanskrit-to-english-learning.csv')

fdist = FreqDist(word_list)
# print(fdist.most_common(15))
most_common_fdist = FreqDist(fdist.most_common(30)).plot(title = 'Word frequency distribution', cumulative = False)


