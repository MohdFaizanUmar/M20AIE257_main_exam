import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from M20AIE257_Q1_MLOPS import test_case_random, test_case_not_random
import argparse
from joblib import dump


digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

parser = argparse.ArgumentParser(
                    prog = 'test',
                    description = 'train a classifier',
                    epilog = 'dummy')

# parser.add_argument('filename')          
parser.add_argument('-c', '--clf_name')      
parser.add_argument('-v', '--random_state')  

args = parser.parse_args()
clf = args.clf_name
random_state = int(args.random_state)



# Split data into 50% train and 50% test subsets
# Q1 a. ensure exact same splitting of the dataset; correspondingly update your code
# Answer. random_state is the argument in train_test_split which ensure exact same plit of the dataset.
print("splitting the data set with random state 42")
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    data, digits.target, test_size=0.5, random_state = random_state
)

if clf == "svm":
    model = svm.SVC()
else:
    model = DecisionTreeClassifier()

model.fit(X_train1,y_train1)

pred = model.predict(X_test1)

model_path = "./models/"
best_model_name = model_path+clf + ".joblib"
dump(model, best_model_name)

a_s = f"test accuracy: {metrics.accuracy_score(y_test1,pred)}"
c_r = f"classification_report: {metrics.classification_report(y_test1,pred)}"
model_saved = f"model saved at {best_model_name}"
ans_list = []

ans_list.append(a_s)
ans_list.append(c_r)
ans_list.append(model_saved)

test_path = f"./results/{clf}_{str(random_state)}.txt"
with open(test_path, 'w') as fp:
    for item in ans_list:
        # write each item on a new line
        fp.write("%s\n" % item)

print(f"Accuracy: {metrics.accuracy_score(y_test1,pred)}")
print(f"classification report: {metrics.classification_report(y_test1,pred)}")