import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import tree, svm, metrics, neighbors
from sklearn.model_selection import train_test_split

file_path = 'buy_computer.csv'
feature = 4

def load_csv():
    csv = np.loadtxt(file_path, dtype=np.str, delimiter=',')
    data = csv[1:, 0:].astype(np.str)
    label = csv[0, 0:feature].astype(np.str)
    return data, label

if __name__ == '__main__':
    data, label = load_csv()
    target = ['No', 'Yes']
    le = LabelEncoder()
    
    for i in range(0, feature+1):
        data[:, i] = le.fit_transform(data[:, i])
    x = data[:, :feature]
    y = data[:, feature]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    print('Answer:{}\n'.format(y_test))
    y_test_predicted = clf.predict(x_test)
    print('Decision Tree')
    print('Predict:', y_test_predicted)

    accuracy = metrics.accuracy_score(y_test, y_test_predicted)
    print('Accuracy:', accuracy)
    
    dot_tree = tree.export_graphviz(clf, out_file=None,
                                feature_names=label,
                                class_names=target,
                                filled=True,
                                rounded=True)
    fp = open('dot_tree.txt', 'w')
    fp.write(dot_tree)
    fp.close
    
    print('\nSVM')
    svc = svm.SVC(kernel='rbf', gamma='auto').fit(x_train, y_train).predict(x_test)
    accuracy = metrics.accuracy_score(y_test, svc)

    print('Predict:', svc)
    print('Accuracy:',accuracy)