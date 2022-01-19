from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import csv


def algorithm_decision(x_train, y_train, x_test, y_test, algorithm, name):
    classifier = DecisionTreeClassifier(criterion=algorithm)
    classifier.fit(x_train, y_train)
    y_prediction = classifier.predict(x_test)
    print("-----------{} Decision tree: -----------------".format(name))
    print("+++++++++++ confusion matrix +++++++++++++++++++")
    print(confusion_matrix(y_test, y_prediction))
    print("+++++++++++ classification result ++++++++++++++")
    print(classification_report(y_test, y_prediction))
    correct = 0
    for i in range(len(y_prediction)):
        if y_test[i] == y_prediction[i]:
            correct += 1
    print("+++++++++++ Accuracy ++++++++++++++++++++")
    print(correct / float(len(y_test)))


def main():

    traindata = csv.reader(open('aps_failure_training_set.csv'))
    train = []
    for row in traindata:
         train.append(list(row))

    for i in range(1, len(train[21])):
        pos = 0.0
        neg = 0.0
        posc = 0
        negc = 0
        for j in range(21, len(train)):
            if train[j][0] == 'pos' and train[j][i] != 'na':
                pos += float(train[j][i])
                posc += 1
            elif train[j][0] == 'neg' and train[j][i] != 'na':
                neg += float(train[j][i])
                negc += 1
        if posc != 0:
            pos = pos / posc
        if negc != 0:
            neg = neg / negc
        for j in range(21, len(train)):
            if train[j][i] == 'na' and train[j][0] == 'neg':
                train[j][i] = str(neg)
            if train[j][i] == 'na' and train[j][0] == 'pos':
                train[j][i] = str(pos)

    y_train = []
    x_train = []
    for o in range(21, len(train)):
        x = []
        for i in range(1, len(train[21])):
            x.append(train[o][i])
        x_train.append(x)

    for m in range(21, len(train)):
        y_train.append(train[m][0])


    testdata = csv.reader(open('aps_failure_test_set.csv'))
    test = []
    for row in testdata:
        test.append(list(row))

    for i in range(1, len(test[21])):
        pos = 0.0
        neg = 0.0
        posc = 0
        negc = 0
        for j in range(21, len(test)):
            if test[j][0] == 'pos' and test[j][i] != 'na':
                pos += float(test[j][i])
                posc += 1
            elif test[j][0] == 'neg' and test[j][i] != 'na':
                neg += float(test[j][i])
                negc += 1
        if posc != 0:
            pos = pos / posc
        if negc != 0:
            neg = neg / negc
        for j in range(21, len(test)):
            if test[j][i] == 'na' and test[j][0] == 'neg':
                test[j][i] = str(neg)
            if test[j][i] == 'na' and test[j][0] == 'pos':
                test[j][i] = str(pos)

    y_test = []
    x_test = []
    for o in range(21, len(test)):
        x = []
        for i in range(1, len(test[21])):
            x.append(test[o][i])
        x_test.append(x)

    for m in range(21, len(test)):
        y_test.append(test[m][0])


    for (algorithm, name) in [("gini", "ID3"), ("entropy",  "C4.5")]:
        algorithm_decision(x_train, y_train, x_test, y_test, algorithm, name)


if __name__ == '__main__':
    main()
