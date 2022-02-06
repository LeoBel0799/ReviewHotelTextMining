from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt


def tfidf (train_X, test_X):
        tf_idf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word', stop_words='english', ngram_range=(1, 5))
        # applying tf idf to training data
        X_train_tf = tf_idf.fit_transform(train_X)  # perche fit e trasform sul train e non sul test
        # applying tf idf to training data
        X_train_tf = tf_idf.transform(train_X)
        # transforming test data into tf-idf matrix
        X_test_tf = tf_idf.transform(test_X)

        feature_names = tf_idf.get_feature_names()
        dense = X_train_tf.todense()
        denselist = dense.tolist()
        df2 = pd.DataFrame(denselist, columns=feature_names)
        print(df2)
        # print(X_train_tf.todense())
        # print(X_test_tf.todense())
        return X_train_tf, X_test_tf




def tf_idf_ngramPlotting (train_X, test_X, train_y, test_y):
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
        pipeline = make_pipeline(StandardScaler(with_mean=False),
                                 LogisticRegression(random_state=250))

        names = ["Nearest Neighbors + TFIDF", "Linear SVM + TFIDF", "Random Forest + TFIDF", "Decision Tree + TFIDF",
                 "Multinomial NB+ TFIDF", "Bagging Classifier + TFIDF"]

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
    plt.show()
