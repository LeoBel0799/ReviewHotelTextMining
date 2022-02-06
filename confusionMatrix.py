import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

def confusionMatrix(test, prediction, name):
    # Plot non-normalized confusion matrix
    matrix = metrics.confusion_matrix(test, prediction)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap=sns.diverging_palette(150, 275, s=80, l=55, n=9),
                xticklabels=["Bad","Neutral","Good"],
                yticklabels=["Bad","Neutral","Good"],
                annot=True,
                fmt='d')
    plt.title(name)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
