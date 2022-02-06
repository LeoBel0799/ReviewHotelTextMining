import re
import string
import sys


import numpy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
numpy.set_printoptions(threshold=sys.maxsize)
import pandas as pd


# TEXT RETRIVE & ELABORATION TEST & TRAIN
# TEXT RETRIEVE
# 1. lower case
# 2. remove punctuation
# 3. remove mention @...
# 4 remove link http:....
# 5. remove emoji

# TEXT ELABORATION
# 1. tokenization
# 2. stop word filtering
# 3. stemming


# reading the data
def preprocessingPhase():
    # reading the data
    test_csv = pd.read_csv('test.csv')
    train_csv = pd.read_csv('BalancedTrain.csv')

    train_X_non = train_csv['reviews.text']  # without preprocessing
    train_y = train_csv['Evaluation']  # only evaluation
    test_X_non = test_csv['reviews.text']
    test_y = test_csv['Evaluation']
    train_X_cleaned = []
    test_X_cleaned = []
    train_X = []  # with preprocessing
    test_X = []
    import re
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)

    # Cleaning Train
    import re
    for i in range(0, len(train_X_non)):
        review = train_X_non[i]
        review = review.lower()
        review = review.translate(str.maketrans('', '', string.punctuation))  # remove puntuaction
        review = re.sub('@[^\s]+', '', review)  # remove mention
        review = re.sub('http[^\s]+', '', review)  # remove link
        review = emoji_pattern.sub(r'', review)  # remove emoji
        train_X_cleaned.append(review)

    # Cleaning Test
    import re
    for i in range(0, len(test_X_non)):
        review = test_X_non[i]
        review = review.lower()
        review = review.translate(str.maketrans('', '', string.punctuation))
        review = re.sub('@[^\s]+', '', review)  # remove mention
        review = re.sub('http[^\s]+', '', review)  # remove link
        review = emoji_pattern.sub(r'', review)  # remove emoji
        test_X_cleaned.append(review)

    lemmatizer = WordNetLemmatizer()
    # Preprocessing Train
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()
    filtered_sent = []
    for i in range(0, len(train_X_cleaned)):
        review = train_X_cleaned[i]
        review = word_tokenize(review)  # tokenization
        for w in review:  # stop word filtering
            if w not in stop_words:
                filtered_sent.append(w)
        review = filtered_sent
        filtered_sent = []
        for w in review:  # stemming
            filtered_sent.append(lemmatizer.lemmatize(w))
        review = filtered_sent
        filtered_sent = []
        review = ' '.join(review)
        train_X.append(review)

    # Preprocessing Test
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()
    filtered_sent = []
    for i in range(0, len(test_X_cleaned)):
        review = test_X_cleaned[i]
        review = word_tokenize(review)  # tokenization
        for w in review:  # stop word filtering
            if w not in stop_words:
                filtered_sent.append(w)
        review = filtered_sent
        filtered_sent = []
        for w in review:  # stemming
            filtered_sent.append(lemmatizer.lemmatize(w))
        review = filtered_sent
        filtered_sent = []
        review = ' '.join(review)  # CON QUESTA UNISCO
        test_X.append(review)

    return train_X, test_X, train_y, test_y


def preprocessing_prediction(review):
    filtered_sent = []
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    rev = []
    test = [review]
    review = test[0]
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    review = review.lower()
    review = review.translate(str.maketrans('', '', string.punctuation))  # remove puntuaction
    review = re.sub('@[^\s]+', '', review)  # remove mention
    review = re.sub('http[^\s]+', '', review)  # remove link
    review = emoji_pattern.sub(r'', review)  # remove emoji

    # processing
    review = word_tokenize(review)  # tokenization
    for w in review:  # stop word filtering
        if w not in stop_words:
            filtered_sent.append(w)
    review = filtered_sent
    for w in review:  # stemming
        filtered_sent.append(lemmatizer.lemmatize(w))
    review = filtered_sent
    review = ' '.join(review)
    rev.append(review)
    return rev
