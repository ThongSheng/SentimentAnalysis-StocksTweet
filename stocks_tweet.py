import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reading in the dataset and splitting the dataset into tweets and sentiment
stock = pd.read_csv("stock_data.csv")
tweets = stock["Text"]
original_sentiment = stock["Sentiment"]

# Cheking if any null values are present
tweets.isnull().any()

# Tokenization 
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\s+', gaps = True)
tweets1 = tweets.apply(lambda x: tokenizer.tokenize(x.lower()))
tweets1.head(10)

# Remove stop words
from nltk.corpus import stopwords
def remove_stopwords(text):
    words = [w for w in text if not w in stopwords.words("english")]
    return words

tweets2 = tweets1.apply(lambda x: remove_stopwords(x))
tweets2.head(10)

# Lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def word_lemmatizer(text):
    lem_text = " ".join([lemmatizer.lemmatize(i) for i in text])
    return lem_text

tweets_final = tweets2.apply(lambda x: word_lemmatizer(x))
tweets_final.head(10)

# Sentiment Analysis
from textblob import TextBlob
def find_pol(text):
    return TextBlob(text).sentiment.polarity

tweets_sentiment = tweets_final.apply(lambda x: find_pol(x))
tweets_sentiment.head(10)

# Data Visualization - Histogram
sns.distplot(tweets_sentiment)
plt.xlabel("Sentiment")
plt.title("Histogram of Tweets Sentiment")

# Data Visualization - Pie Chart
negative = tweets_sentiment[(tweets_sentiment < 0) & (tweets_sentiment >= -1)]
positive = tweets_sentiment[(tweets_sentiment > 0) & (tweets_sentiment <= 1)]
neutral = tweets_sentiment[tweets_sentiment == 0]
plt.pie([len(positive), len(negative), len(neutral)], labels = ["Positive", "Negative", "Neutral"])
plt.title("Pie Chart of Tweets Sentiment")

# Comparing results
from sklearn.metrics import mean_squared_error
mean_squared_error(original_sentiment, tweets_sentiment)
# Error is 0.9483855399480924.