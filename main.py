from sklearn.feature_extraction.text import TfidfVectorizer

from classificationReport import classificationReport
from classifierComparison import classifierComparison
from classifiers import multinomialnb, svm, randomForest, decisionTree, knnclassifier, baggingClassifier
from confusionMatrix import confusionMatrix
from importingandbalancing import importingandbalancing
from textpreprocessing import preprocessingPhase, preprocessing_prediction
from tfidf import tfidf, tf_idf_ngramPlotting

importingandbalancing()
train_X, test_X, train_y, test_y = preprocessingPhase()
X_train_tf, X_test_tf = tfidf(train_X, test_X)

MultinomialNB = multinomialnb(X_train_tf, train_y)
predictionMultinomial = MultinomialNB.predict(X_test_tf)
confusionMatrix(test_y, predictionMultinomial, name ='Confusion Matrix MultinomialNB')
classificationReport(test_y ,predictionMultinomial)

SVM = svm (X_train_tf, train_y)
predictionSVM = SVM.predict(X_test_tf)
confusionMatrix(test_y, predictionSVM, name ='Confusion Matrix Support Vector Machine')
classificationReport(test_y ,predictionSVM)

randomF = randomForest (X_train_tf, train_y)
predictionRandomF = randomF.predict(X_test_tf)
confusionMatrix(test_y, predictionRandomF, name ='Confusion Matrix Random Forest')
classificationReport(test_y ,predictionRandomF)

decision = decisionTree (X_train_tf, train_y)
predictionDecisionTree = decision.predict(X_test_tf)
confusionMatrix(test_y, predictionDecisionTree, name ='Confusion Matrix Decision Tree')
classificationReport(test_y ,predictionDecisionTree)

knn = knnclassifier (X_train_tf, train_y)
predictionKNN = knn.predict(X_test_tf)
confusionMatrix(test_y, predictionKNN, name ='Confusion Matrix KNN')
classificationReport(test_y ,predictionKNN)

bagging = baggingClassifier(X_train_tf, train_y)
predictionBagging = bagging.predict(X_test_tf)
confusionMatrix(test_y, predictionBagging, name ='Confusion Matrix Bagging')
classificationReport(test_y ,predictionBagging)

classifierComparison(X_train_tf, train_y,X_test_tf, test_y)

tf_idf_ngramPlotting(train_X, test_X, train_y, test_y)


#TO CHECK
# review = input('\n\nInsert a review to classify: ')
# preprocessing_prediction(review)
# tf_idf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word', stop_words='english')
# rev_process = tf_idf.transform(review)
# print("Prediction with MultinomialDB")
# MNB = MultinomialNB.predict(rev_process)[0]
