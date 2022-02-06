from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def multinomialnb(X_train_tf, train_y):
    print("\nBuilding Mulinomial Naive Bayes Classifier:")
    multinomial = MultinomialNB()
    multinomial.fit(X_train_tf, train_y)
    return multinomial

def svm (X_train_tf, train_y):
    print("\nBuilding SVM Classifier:")
    svmclf = SVC(kernel='linear')
    svmclf.fit(X_train_tf, train_y)
    return svmclf

def randomForest (X_train_tf, train_y):
    print("\nBuilding Random Forest Classifier:")
    forest = RandomForestClassifier(max_depth=5, n_estimators=100)
    forest.fit(X_train_tf, train_y)
    return forest

def decisionTree (X_train_tf, train_y):
    print("\nBuilding Decision Tree Classifier:")
    decision = DecisionTreeClassifier(max_depth=5, random_state = 5)
    decision.fit(X_train_tf, train_y)
    return decision

def knnclassifier (X_train_tf, train_y):
    print("\nBuilding KNN Classifier:")
    knn = KNeighborsClassifier(n_neighbors=75)
    knn.fit(X_train_tf, train_y)
    return knn

def baggingClassifier (X_train_tf, train_y):
    print("\nBuilding Bagging Classifier:")
    pipeline = make_pipeline(StandardScaler(with_mean=False),
                             LogisticRegression(random_state=250))

    bgclassifier = BaggingClassifier(base_estimator=pipeline, n_estimators=100,
                                     max_features=10,
                                     max_samples=100,
                                     random_state=250)

    bgclassifier.fit(X_train_tf, train_y)
    return bgclassifier


