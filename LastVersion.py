# importing libraries
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import string
import sys
import numpy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import seaborn as sns
from csv import reader
import matplotlib.pyplot as plt
import re
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from time import process_time
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA






numpy.set_printoptions(threshold=sys.maxsize)

# SPLIT & BALANCE
# split train and test and do the balance on the train
# traib 75% test 25%




# split the data into train and test set
# reading dataset
readCsv = pd.read_csv("csv_files/Datafiniti_Hotel_Reviews_Jun19_3.csv")

train, test = train_test_split(readCsv, test_size=0.25, random_state=0)
# save the data
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)

# PLOT 1
plt.title("Class of imbalanced training Set")
# read a tips.csv file from seaborn library
df = pd.read_csv("csv_files/train.csv")
# count plot along x axis
sns.countplot(x="Evaluation", data=df, palette="magma")
# Show the plot
plt.show()

# do rebalncing training set
class_0 = train[train['Evaluation'] == "Good"]
class_1 = train[train['Evaluation'] == "Bad"]
class_2 = train[train['Evaluation'] == "Neutral"]
class_count_0, class_count_1, class_count_2 = train['Evaluation'].value_counts()
class_0_under = class_0.sample(class_count_1)
test_under = pd.concat([class_0_under, class_1, class_2], axis=0)

# print("total class of Good and Bad: \n",test_under['Evaluation'].value_counts())# plot the count after under-sampeling


test_under.to_csv("BalancedTrain.csv", index=None)
BalancedTrain = pd.read_csv('csv_files/BalancedTrain.csv')

# PLOT2
plt.title("Class of balanced training Set")
# read a tips.csv file from seaborn library
df = pd.read_csv("csv_files/BalancedTrain.csv")
# count plot along x axis
sns.countplot(x="Evaluation", data=df, palette="magma")
# Show the plot
plt.show()

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
test_csv = pd.read_csv('csv_files/test.csv')
train_csv = pd.read_csv('csv_files/BalancedTrain.csv')

train_X_non = train_csv['reviews.text']  # without preprocessing
train_y = train_csv['Evaluation']  # only evaluation
test_X_non = test_csv['reviews.text']
test_y = test_csv['Evaluation']
train_X_cleaned = []
test_X_cleaned = []
train_X = []  # with preprocessing
test_X = []

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

# Cleaning Train

for i in range(0, len(train_X_non)):
    review = train_X_non[i]
    review = review.lower()
    review = review.translate(str.maketrans('', '', string.punctuation))  # remove puntuaction
    review = re.sub('@[^\s]+', '', review)  # remove mention
    review = re.sub('http[^\s]+', '', review)  # remove link
    review = emoji_pattern.sub(r'', review)  # remove emoji
    train_X_cleaned.append(review)

# Cleaning Test

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

# TF-IDF


tf_idf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word', stop_words='english', ngram_range=(1, 5))
# applying tf idf to training data
X_train_tf = tf_idf.fit_transform(train_X)  # perche fit e trasform sul train e non sul test
# applying tf idf to training data
X_train_tf = tf_idf.transform(train_X)
# transforming test data into tf-idf matrix
X_test_tf = tf_idf.transform(test_X)

NUMBER_OF_CLUSTERS = 3
km = KMeans(
    n_clusters=NUMBER_OF_CLUSTERS,
    init='k-means++',
    max_iter=500)
km.fit(X_train_tf)
# First: for every document we get its corresponding cluster
clusters = km.predict(X_train_tf)

# We train the PCA on the dense version of the tf-idf.
pca = PCA(n_components=2)
two_dim = pca.fit_transform(X_train_tf.todense())

scatter_x = two_dim[:, 0] # first principle component
scatter_y = two_dim[:, 1] # second principle component


plt.style.use('ggplot')

fig, ax = plt.subplots()
fig.set_size_inches(20,10)

# color map for NUMBER_OF_CLUSTERS we have
cmap = {0: 'green', 1: 'blue', 2: 'red'}

# group by clusters and scatter plot every cluster
# with a colour and a label
for group in np.unique(clusters):
    ix = np.where(clusters == group)
    ax.scatter(scatter_x[ix], scatter_y[ix], c=cmap[group], label=group)

ax.legend()
plt.xlabel("PCA 0")
plt.ylabel("PCA 1")
plt.show()

order_centroids = km.cluster_centers_.argsort()[:, ::-1]

terms = tf_idf.get_feature_names()
for i in range(3):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :50]:
        print(' %s' % terms[ind], end='')
    print()




# MULTINOMIAL NAIVE BAYES CLASSIFIER + TFIDF


naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_tf, train_y)  # faccio il train con le recensioni + l'evaluation
y_pred = naive_bayes_classifier.predict(X_test_tf)  # faccio la predizione delle recensioni test
np.set_printoptions(precision=2)
class_names = ['Bad', 'Neutral', 'Good']

titles_options = [
    ("Confusion matrix, without normalization -- MULTINOMIAL DB", None),
    ("Normalized confusion matrix -- MULTINOMIAL DB", "true"),
]
# for title, normalize in titles_options:
#   disp = ConfusionMatrixDisplay.from_estimator(
#      naive_bayes_classifier,
#      X_test_tf,
#      test_y,
#      display_labels=class_names,
#      cmap=plt.cm.Blues,
#      normalize=normalize,
#  )
#  disp.ax_.set_title(title)


# plt.show()

results = confusion_matrix(test_y, y_pred)
error = zero_one_loss(test_y, y_pred)

FP = results.sum(axis=0) - np.diag(results)
FN = results.sum(axis=1) - np.diag(results)
TP = np.diag(results)
TN = results.sum() - (FP + FN + TP)
prec = TP / TP + FP
sens = TP / prec

print('\n Time Processing: \n', process_time())
print('\n Zero-one classification loss: \n', error)
print('\n True Positive: \n', TP)
print('\n True Negative: \n', TN)
print('\n False Positive: \n', FP)
print('\n False Negative: \n', FN)
print('\n Sensitivity: \n', sens)

print('\n The Classification report:\n', classification_report(test_y, y_pred, digits=2))
print('Accuracy:', accuracy_score(test_y, y_pred))

# SVM + TFIDF
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn import svm
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from time import process_time

# Create a svm Classifier
clf = svm.SVC(kernel='linear')  # Linear Kernel

# Train the model using the training sets
clf.fit(X_train_tf, train_y)

# Predict the response for test dataset
y_pred = clf.predict(X_test_tf)
np.set_printoptions(precision=2)
class_names = ['Bad', 'Neutral', 'Good']

titles_options = [
    ("Confusion matrix, without normalization -- SVM", None),
    ("Normalized confusion matrix -- SVM", "true"),
]
# for title, normalize in titles_options:
#   disp = ConfusionMatrixDisplay.from_estimator(
#      naive_bayes_classifier,
#      X_test_tf,
#      test_y,
#      display_labels=class_names,
#      cmap=plt.cm.Blues,
#      normalize=normalize,
#  )
#  disp.ax_.set_title(title)


# plt.show()

results = confusion_matrix(test_y, y_pred)
error = zero_one_loss(test_y, y_pred)

FP = results.sum(axis=0) - np.diag(results)
FN = results.sum(axis=1) - np.diag(results)
TP = np.diag(results)
TN = results.sum() - (FP + FN + TP)
prec = TP / TP + FP
sens = TP / prec

print('\n Time Processing: \n', process_time())
print('\n Zero-one classification loss: \n', error)
print('\n True Positive: \n', TP)
print('\n True Negative: \n', TN)
print('\n False Positive: \n', FP)
print('\n False Negative: \n', FN)
print('\n Sensitivity: \n', sens)

print('\n The Classification report:\n', classification_report(test_y, y_pred, digits=2))
print('Accuracy:', accuracy_score(test_y, y_pred))

# RANDOM FOREST + TFIDF
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from time import process_time

forest = RandomForestClassifier(max_depth=5, n_estimators=100)
forest.fit(X_train_tf, train_y)
y_pred = forest.predict(X_test_tf)
np.set_printoptions(precision=2)
class_names = ['Bad', 'Neutral', 'Good']

titles_options = [
    ("Confusion matrix, without normalization -- RANDOM FOREST", None),
    ("Normalized confusion matrix -- RANDOM FOREST", "true"),
]
# for title, normalize in titles_options:
#   disp = ConfusionMatrixDisplay.from_estimator(
#      naive_bayes_classifier,
#      X_test_tf,
#      test_y,
#      display_labels=class_names,
#      cmap=plt.cm.Blues,
#      normalize=normalize,
#  )
#  disp.ax_.set_title(title)


# plt.show()

results = confusion_matrix(test_y, y_pred)
error = zero_one_loss(test_y, y_pred)

FP = results.sum(axis=0) - np.diag(results)
FN = results.sum(axis=1) - np.diag(results)
TP = np.diag(results)
TN = results.sum() - (FP + FN + TP)
prec = TP / TP + FP
sens = TP / prec

print('\n Time Processing: \n', process_time())
print('\n Zero-one classification loss: \n', error)
print('\n True Positive: \n', TP)
print('\n True Negative: \n', TN)
print('\n False Positive: \n', FP)
print('\n False Negative: \n', FN)
print('\n Sensitivity: \n', sens)

print('\n The Classification report:\n', classification_report(test_y, y_pred, digits=2))
print('Accuracy:', accuracy_score(test_y, y_pred))

# DECISION TREE + TFIDF
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from time import process_time

decison_gini = DecisionTreeClassifier(random_state=5, max_depth=10)
decison_gini.fit(X_train_tf, train_y)
y_pred = decison_gini.predict(X_test_tf)

np.set_printoptions(precision=2)
class_names = ['Bad', 'Neutral', 'Good']

titles_options = [
    ("Confusion matrix, without normalization -- DECISION TREE", None),
    ("Normalized confusion matrix -- DECISION TREE", "true"),
]
# for title, normalize in titles_options:
#   disp = ConfusionMatrixDisplay.from_estimator(
#      naive_bayes_classifier,
#      X_test_tf,
#      test_y,
#      display_labels=class_names,
#      cmap=plt.cm.Blues,
#      normalize=normalize,
#  )
#  disp.ax_.set_title(title)


# plt.show()

results = confusion_matrix(test_y, y_pred)
error = zero_one_loss(test_y, y_pred)

FP = results.sum(axis=0) - np.diag(results)
FN = results.sum(axis=1) - np.diag(results)
TP = np.diag(results)
TN = results.sum() - (FP + FN + TP)
prec = TP / TP + FP
sens = TP / prec

print('\n Time Processing: \n', process_time())
print('\n Zero-one classification loss: \n', error)
print('\n True Positive: \n', TP)
print('\n True Negative: \n', TN)
print('\n False Positive: \n', FP)
print('\n False Negative: \n', FN)
print('\n Sensitivity: \n', sens)

print('\n The Classification report:\n', classification_report(test_y, y_pred, digits=2))
print('Accuracy:', accuracy_score(test_y, y_pred))

# KNN + TFIDF

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from time import process_time

knn = KNeighborsClassifier(n_neighbors=75)
knn.fit(X_train_tf, train_y)
y_pred = knn.predict(X_test_tf)
np.set_printoptions(precision=2)
class_names = ['Bad', 'Neutral', 'Good']

titles_options = [
    ("Confusion matrix, without normalization -- KNN", None),
    ("Normalized confusion matrix -- KNN", "true"),
]
# for title, normalize in titles_options:
#   disp = ConfusionMatrixDisplay.from_estimator(
#      naive_bayes_classifier,
#      X_test_tf,
#      test_y,
#      display_labels=class_names,
#      cmap=plt.cm.Blues,
#      normalize=normalize,
#  )
#  disp.ax_.set_title(title)


# plt.show()

results = confusion_matrix(test_y, y_pred)
error = zero_one_loss(test_y, y_pred)

FP = results.sum(axis=0) - np.diag(results)
FN = results.sum(axis=1) - np.diag(results)
TP = np.diag(results)
TN = results.sum() - (FP + FN + TP)
prec = TP / TP + FP
sens = TP / prec

print('\n Time Processing: \n', process_time())
print('\n Zero-one classification loss: \n', error)
print('\n True Positive: \n', TP)
print('\n True Negative: \n', TN)
print('\n False Positive: \n', FP)
print('\n False Negative: \n', FN)
print('\n Sensitivity: \n', sens)

print('\n The Classification report:\n', classification_report(test_y, y_pred, digits=2))
print('Accuracy:', accuracy_score(test_y, y_pred))

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV

pipeline = make_pipeline(StandardScaler(with_mean=False),
                         LogisticRegression(random_state=250))

bgclassifier = BaggingClassifier(base_estimator=pipeline, n_estimators=100,
                                 max_features=10,
                                 max_samples=100,
                                 random_state=250)

bgclassifier.fit(X_train_tf, train_y)
y_pred = bgclassifier.predict(X_test_tf)

np.set_printoptions(precision=2)
class_names = ['Bad', 'Neutral', 'Good']

titles_options = [
    ("Confusion matrix, without normalization -- BAGGING", None),
    ("Normalized confusion matrix -- BAGGING", "true"),
]
# for title, normalize in titles_options:
#   disp = ConfusionMatrixDisplay.from_estimator(
#      naive_bayes_classifier,
#      X_test_tf,
#      test_y,
#      display_labels=class_names,
#      cmap=plt.cm.Blues,
#      normalize=normalize,
#  )
#  disp.ax_.set_title(title)


# plt.show()

results = confusion_matrix(test_y, y_pred)
error = zero_one_loss(test_y, y_pred)

FP = results.sum(axis=0) - np.diag(results)
FN = results.sum(axis=1) - np.diag(results)
TP = np.diag(results)
TN = results.sum() - (FP + FN + TP)
prec = TP / TP + FP
sens = TP / prec

print('\n Time Processing: \n', process_time())
print('\n Zero-one classification loss: \n', error)
print('\n True Positive: \n', TP)
print('\n True Negative: \n', TN)
print('\n False Positive: \n', FP)
print('\n False Negative: \n', FN)
print('\n Sensitivity: \n', sens)

print('\n The Classification report:\n', classification_report(test_y, y_pred, digits=2))
print('Accuracy:', accuracy_score(test_y, y_pred))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

names = ["Nearest Neighbors + TFIDF", "Linear SVM + TFIDF", "Random Forest + TFIDF", "Decision Tree + TFIDF",
         "Multinomial NB+ TFIDF", "Bagging Classifier + TFIDF"]

pipeline = make_pipeline(StandardScaler(with_mean=False),
                         LogisticRegression(random_state=250))

classifiers = [
    KNeighborsClassifier(75),
    SVC(kernel="linear"),
    RandomForestClassifier(max_depth=5, n_estimators=100),
    DecisionTreeClassifier(random_state=5, max_depth=10),
    MultinomialNB(),
    BaggingClassifier(base_estimator=pipeline, n_estimators=100,
                      max_features=10,
                      max_samples=100,
                      random_state=250)
]

# iterate on classifiers
scores = []
for name, clfs in zip(names, classifiers):
    clfs.fit(X_train_tf, train_y)
    score = clfs.score(X_test_tf, test_y)
    scores.append(score)


df = pd.DataFrame()
df['Classifier_Name'] = names
df['Accuracy score'] = scores
df


sns.set(style="whitegrid")
ax = sns.barplot(x="Accuracy score", y="Classifier_Name", data=df)
plt.show()


test2 = []
# doing a test prediction
test = ['bad orrible rat']
review = test[0]
# cleaning
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
filtered_sent = []
for w in review:  # stemming
    filtered_sent.append(lemmatizer.lemmatize(w))
review = filtered_sent
filtered_sent = []
review = ' '.join(review)
test2.append(review)

print(test2)
X_test_tf = tf_idf.transform(test2)

print("Prediction with MultinomialDB")
MNB = naive_bayes_classifier.predict(X_test_tf)[0]
print(MNB)

# PLOT THE DIFFERENCE BETWEEN N-GRAM
i = 1
acc = 'n-grams = '
df = pd.DataFrame()
while i <= 5:
    # TF-IDF
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer

    tf_idf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word', stop_words='english',
                             ngram_range=(1, i))
    # applying tf idf to training data
    X_train_tf = tf_idf.fit_transform(train_X)  # perche fit e trasform sul train e non sul test
    # applying tf idf to training data
    X_train_tf = tf_idf.transform(train_X)
    # transforming test data into tf-idf matrix
    X_test_tf = tf_idf.transform(test_X)
    classifiers = [
        KNeighborsClassifier(75),
        SVC(kernel="linear"),
        RandomForestClassifier(max_depth=5, n_estimators=100),
        DecisionTreeClassifier(random_state=5, max_depth=10),
        MultinomialNB(),
        BaggingClassifier(base_estimator=pipeline, n_estimators=100,
                          max_features=10,
                          max_samples=100,
                          random_state=250)
    ]
    # iterate on classifiers
    scores = []
    for name, clfs in zip(names, classifiers):
        clfs.fit(X_train_tf, train_y)
        score = clfs.score(X_test_tf, test_y)
        scores.append(score)

    df['Classifier_Name'] = names
    newColumn = acc + str(i)
    df[newColumn] = scores
    i = i + 1

    # sns.set(style="whitegrid")
    # ax = sns.barplot(x="Accuracy score", y="Classifier_Name", data=df)
    # plt.show()

# print(df)
sns.color_palette("mako")
dfm = df.melt('Classifier_Name', var_name='cols', value_name='vals')
# print(dfm)
g = sns.catplot(x="cols", y="vals", hue='Classifier_Name', data=dfm, kind='point',
                palette=sns.color_palette(['green', 'blue', 'red', 'yellow', 'purple', 'black']))
g.fig.set_size_inches(15, 5)

# SELECT HOTEL TO SHOW
df = pd.read_csv("Datafiniti_Hotel_Reviews_Jun19_2.csv")
df
nameHotel = "Hyatt House Seattle/Downtown"
# creo un dataframe con le informazioni
dfHotel = df.loc[df['name'] == nameHotel]
dfapp = dfHotel[["reviews.text", "name", "reviews.date", "Evaluation"]]
dfapp["class"] = ""
dfapp["valueC"] = ""
dfapp["valueR"] = ""
dfapp

# processing delle review
for index, row in dfapp.iterrows():
    review = row["reviews.text"]
    test2 = []
    # cleaning
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
    filtered_sent = []
    for w in review:  # stemming
        filtered_sent.append(lemmatizer.lemmatize(w))
    review = filtered_sent
    filtered_sent = []
    review = ' '.join(review)
    test2.append(review)
    X_test_tf = tf_idf.transform(test2)
    MNB = naive_bayes_classifier.predict(X_test_tf)[0]
    row["class"] = MNB

# convert format of the date
import datetime

new_format = "%Y-%m-%d"
for index, row in dfapp.iterrows():
    d1 = datetime.datetime.strptime(row["reviews.date"], "%Y-%m-%dT%H:%M:%S.%fZ")
    row["reviews.date"] = d1.strftime(new_format)

for index, row in dfapp.iterrows():
    if (row["class"] == "Good"):
        row["valueC"] = 1
    if (row["class"] == "Neutral"):
        row["valueC"] = 0
    if (row["class"] == "Bad"):
        row["valueC"] = -1

for index, row in dfapp.iterrows():
    if (row["Evaluation"] == "Good"):
        row["valueR"] = 1
    if (row["Evaluation"] == "Neutral"):
        row["valueR"] = 0
    if (row["Evaluation"] == "Bad"):
        row["valueR"] = -1

dfapp

# plot dell'andamento
Yearwise = dfapp.groupby(by=('reviews.date')).sum()['valueC']
plt.figure(figsize=(14, 10))
Yearwise.plot()

Yearwise = dfapp.groupby(by=('reviews.date')).sum()['valueR']
plt.figure(figsize=(14, 10))
Yearwise.plot()




