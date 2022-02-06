from time import process_time
from sklearn.metrics import confusion_matrix, zero_one_loss, classification_report, accuracy_score
import numpy


def classificationReport(test_y, y_pred):
    results = confusion_matrix(test_y, y_pred)
    error = zero_one_loss(test_y, y_pred)
    lerror = round(error,2)

    FP = results.sum(axis=0) - numpy.diag(results)
    FN = results.sum(axis=1) - numpy.diag(results)
    TP = numpy.diag(results)
    TN = results.sum() - (FP + FN + TP)
    prec = TP / TP + FP
    sens = TP / prec
    lsens = numpy.round(sens,2)

    print('\n Time Processing: \n', process_time())
    print('\n Zero-one classification loss: \n', lerror)
    print('\n True Positive: \n', TP)
    print('\n True Negative: \n', TN)
    print('\n False Positive: \n', FP)
    print('\n False Negative: \n', FN)
    print('\n Sensitivity: \n', lsens)

    print('\n The Classification report:\n', classification_report(test_y, y_pred, digits=2))
    accuracy = accuracy_score(test_y, y_pred)
    laccuracy = numpy.round(accuracy,2)
    print('Accuracy:', laccuracy)
