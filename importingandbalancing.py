from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def importingandbalancing():

    #split the data into train and test set
    #reading dataset
    readCsv = pd.read_csv("csv_files/Datafiniti_Hotel_Reviews_Jun19_3.csv")


    train,test = train_test_split(readCsv, test_size=0.25, random_state=0)
    #save the data
    train.to_csv('train.csv',index=False)
    test.to_csv('test.csv',index=False)

    #PLOT 1
    plt.title("Class of imbalanced training Set")
    # read a tips.csv file from seaborn library
    df = pd.read_csv("csv_files/train.csv")
    # count plot along x axis
    sns.countplot(x="Evaluation", data = df , palette="magma")
    # Show the plot
    plt.show()


    #do rebalncing training set
    class_0 = train[train['Evaluation'] == "Good"]
    class_1 = train[train['Evaluation'] == "Bad"]
    class_2 = train[train['Evaluation'] == "Neutral"]
    class_count_0, class_count_1, class_count_2 = train['Evaluation'].value_counts()
    class_0_under = class_0.sample(class_count_1)
    test_under = pd.concat([class_0_under, class_1, class_2], axis=0)


    #print("total class of Good and Bad: \n",test_under['Evaluation'].value_counts())# plot the count after under-sampeling


    test_under.to_csv("BalancedTrain.csv", index=None)
    BalancedTrain = pd.read_csv('csv_files/BalancedTrain.csv')

    #PLOT2
    plt.title("Class of balanced training Set")
    # read a tips.csv file from seaborn library
    df = pd.read_csv("csv_files/BalancedTrain.csv")
    # count plot along x axis
    sns.countplot(x="Evaluation", data = df , palette="magma")
    # Show the plot
    plt.show()