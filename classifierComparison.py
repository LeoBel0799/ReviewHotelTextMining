from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def classifierComparison(X_train_tf, train_y,X_test_tf, test_y):

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

    scores = []
    for name, clfs in zip(names, classifiers):
        clfs.fit(X_train_tf, train_y)
        score = clfs.score(X_test_tf, test_y)
        scores.append(score)

    df = pd.DataFrame()
    df['Classifier_Name'] = names
    df['Accuracy score'] = scores

    sns.set(style="whitegrid")
    ax = sns.barplot(x="Accuracy score", y="Classifier_Name", data=df)
    plt.show()
