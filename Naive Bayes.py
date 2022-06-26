# Importing the dependencies
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE, RandomOverSampler
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, matthews_corrcoef, classification_report
from numpy.random import seed
import random


seed(1)
random.seed(2)
random_state = 42


# Loading the dataset into pandas
from sklearn.preprocessing import StandardScaler

creditCard_data = pd.read_csv('creditcard.csv')

# First five rows of the dataset
print(creditCard_data.head())

# Last five rows of the dataset
print(creditCard_data.tail())

# Dataset Information
print(creditCard_data.info())

# Checking the number of missing values in each column
print(creditCard_data.isnull().sum())

# Distribution of legal & fraud transactions
print(creditCard_data['Class'].value_counts())

# Separating the data for the analysis
legit = creditCard_data[creditCard_data.Class == 0]
fraud = creditCard_data[creditCard_data.Class == 1]

print(legit.shape)
print(fraud.shape)

# statistical amount of the data
print(legit.Amount.describe())
print(fraud.Amount.describe())

# compare the values of both transactions
print(creditCard_data.groupby('Class').mean())

# Balancing the dat by the under sampling method
# Building a sample dataset containing similar distribution of legit and fraud transactions
# Take random sample of the legit transactions
legit_sample = legit.sample(n=492)
print(legit_sample)

# Concatenating the legit_sample and the fraud dataset
new_dataset = pd.concat([legit_sample, fraud], axis=0)
new_dataset_smot = pd.concat([legit, fraud], axis=0)

print(new_dataset.head())
print(new_dataset.tail())
print(new_dataset['Class'].value_counts())

# mean fo the new dataset to check if we have a good sample
print(new_dataset.groupby('Class').mean())

# Splitting the dataset into Features & Target
features = new_dataset.drop(columns='Class', axis=1)
features_smot = new_dataset_smot.drop(columns='Class', axis=1)

target = new_dataset['Class']
target_smot = new_dataset_smot['Class']

print(features)
print(target)

# Splitting the data into Training data & Testing data
features_train, features_test, target_train, target_test = train_test_split(features, target,
                                                                            test_size=0.2, stratify=target,
                                                                            random_state=random_state)

features_train_smot_over, features_test_smot_over, target_train_smot_over, target_test_smot_over = train_test_split(
    features_smot,
    target_smot,
    test_size=0.2,
    stratify=target_smot,
    random_state=random_state)

# SMOTE Resampling
smote = SMOTE(random_state=random_state)
features_train_smot, target_train_smot = smote.fit_resample(features_train_smot_over, target_train_smot_over)

# Random Over Sampling Technique
overSample = RandomOverSampler(sampling_strategy='minority')
features_train_over, target_train_over = overSample.fit_resample(features_train_smot_over, target_train_smot_over)

# Naive Bayes Model
NBModel = GaussianNB()

# Predicting the test set results
NBModel.fit(features_train, target_train)
target_predict = NBModel.predict(features_test)

NBModel.fit(features_train_smot, target_train_smot)
target_predict_smot = NBModel.predict(features_test_smot_over)

NBModel.fit(features_train_over, target_train_over)
target_predict_over = NBModel.predict(features_test_smot_over)

# Accuracy on training set
print("Training Accuracy Under Sampling is: ", NBModel.score(features_train, target_train))
print("Training Accuracy SMOT is: ", NBModel.score(features_train_smot, target_train_smot))
print("Training Accuracy Random Over Sampling is: ", NBModel.score(features_train_over, target_train_over))

# Accuracy on testing set
print("Testing Accuracy Under sampling is: ", NBModel.score(features_test, target_predict))
print("Testing Accuracy SMOT is: ", NBModel.score(features_test_smot_over, target_predict_smot))
print("Testing Accuracy SMOT is: ", NBModel.score(features_test_smot_over, target_predict_over))


# scoring Naive Bayes Under Sampling
NB_accuracy_score = accuracy_score(target_test, target_predict)
NB_precision_score = precision_score(target_test, target_predict)
NB_recall_score = recall_score(target_test, target_predict)
NB_f1_score = f1_score(target_test, target_predict)
NB_MCC = matthews_corrcoef(target_test, target_predict)

# printing metrics for Under Sampling
print("")
print("Naive Bayes Under Sampling")
print("Scores")
print("Accuracy -->", NB_accuracy_score)
print("Precision -->", NB_precision_score)
print("Recall -->", NB_recall_score)
print("F1 -->", NB_f1_score)
print("MCC -->", NB_MCC)
print(metrics.classification_report(target_test, target_predict))

# Confusion Matrix for Under Sampling/
LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(target_test, target_predict)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix Under Sampling")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

# scoring Naive Bayes SMOT
NB_accuracy_score_smot = accuracy_score(target_test_smot_over, target_predict_smot)
NB_precision_score_smot = precision_score(target_test_smot_over, target_predict_smot)
NB_recall_score_smot = recall_score(target_test_smot_over, target_predict_smot)
NB_f1_score_smot = f1_score(target_test_smot_over, target_predict_smot)
NB_MCC_smot = matthews_corrcoef(target_test_smot_over, target_predict_smot)

# printing metrics for SMOT
print("")
print("Naive Bayes SMOT")
print("Scores")
print("Accuracy -->", NB_accuracy_score_smot)
print("Precision -->", NB_precision_score_smot)
print("Recall -->", NB_recall_score_smot)
print("F1 -->", NB_f1_score_smot)
print("MCC -->", NB_MCC_smot)
print(metrics.classification_report(target_test_smot_over, target_predict_smot))

# Confusion Matrix for SMOT
LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(target_test_smot_over, target_predict_smot)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix SMOT")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# scoring Naive Bayes Over Sampling
NB_accuracy_score_over = accuracy_score(target_test_smot_over, target_predict_over)
NB_precision_score_over = precision_score(target_test_smot_over, target_predict_over)
NB_recall_score_over = recall_score(target_test_smot_over, target_predict_over)
NB_f1_score_over = f1_score(target_test_smot_over, target_predict_over)
NB_MCC_over = matthews_corrcoef(target_test_smot_over, target_predict_over)

# printing metrics for SMOT
print("")
print("Naive Bayes Random Over Sampling")
print("Scores")
print("Accuracy -->", NB_accuracy_score_over)
print("Precision -->", NB_precision_score_over)
print("Recall -->", NB_recall_score_over)
print("F1 -->", NB_f1_score_over)
print("MCC -->", NB_MCC_over)
print(metrics.classification_report(target_test_smot_over, target_predict_over))

# Confusion Matrix for SMOT
LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(target_test_smot_over, target_predict_smot)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix SMOT")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

# Cross validation for SMOT
accuracies = 0
precisions = 0
recalls = 0
f1_scores = 0
n_splits = 5
cv = StratifiedKFold(n_splits=n_splits)
for train_idx, test_idx in cv.split(features_smot, target_smot):
    features_train, target_train = features_smot.iloc[train_idx], target_smot.iloc[train_idx]
    features_test, target_test = features_smot.iloc[test_idx], target_smot.iloc[test_idx]
    features_train, target_train = smote.fit_resample(features_train, target_train)
    NBModel.fit(features_train, target_train)
    target_prediction = NBModel.predict(features_test)
    print(f'Accuracy: {NBModel.score(features_test, target_test)}')
    print(f'Precision: {precision_score(target_test, target_prediction)}')
    accuracies += NBModel.score(features_test, target_test)
    precisions += precision_score(target_test, target_prediction)
    recalls += recall_score(target_test, target_prediction)
    f1_scores += f1_score(target_test, target_prediction)
    print("SMOTE Classification Report -->")
    print(classification_report(target_test, target_prediction))

print("SMOTE Average Accuracy", (accuracies / n_splits) * 100)
print("SMOTE Average Precision", (precisions / n_splits) * 100)
print("SMOTE Average Recall", (recalls / n_splits) * 100)
print("SMOTE Average F1 Score", (f1_scores / n_splits) * 100)

# Cross validation for over sampling
accuracies = 0
precisions = 0
recalls = 0
f1_scores = 0
cv = StratifiedKFold(n_splits=n_splits)
for train_idx, test_idx in cv.split(features_smot, target_smot):
    features_train, target_train = features_smot.iloc[train_idx], target_smot.iloc[train_idx]
    features_test, target_test = features_smot.iloc[test_idx], target_smot.iloc[test_idx]
    features_train, target_train = overSample.fit_resample(features_train, target_train)
    NBModel.fit(features_train, target_train)
    target_prediction = NBModel.predict(features_test)
    print(f'Accuracy: {NBModel.score(features_test, target_test)}')
    print(f'Precision: {precision_score(target_test, target_prediction)}')
    accuracies += NBModel.score(features_test, target_test)
    precisions += precision_score(target_test, target_prediction)
    recalls += recall_score(target_test, target_prediction)
    f1_scores += f1_score(target_test, target_prediction)
    print("Over Sample Classification Report -->")
    print(classification_report(target_test, target_prediction))

print("Random Over Sampling Average Accuracy", (accuracies / n_splits) * 100)
print("Random Over Sampling Average Precision", (precisions / n_splits) * 100)
print("Random Over Sampling Average Recall", (recalls / n_splits) * 100)
print("Random Over Sampling Average F1 Score", (f1_scores / n_splits) * 100)

# Cross validation for under sampling
accuracies = 0
precisions = 0
recalls = 0
f1_scores = 0
cv = StratifiedKFold(n_splits=n_splits)
for train_idx, test_idx in cv.split(features, target):
    features_train, target_train = features.iloc[train_idx], target.iloc[train_idx]
    features_test, target_test = features.iloc[test_idx], target.iloc[test_idx]
    NBModel.fit(features_train, target_train)
    target_prediction = NBModel.predict(features_test)
    print(f'Accuracy: {NBModel.score(features_test, target_test)}')
    print(f'Precision: {precision_score(target_test, target_prediction)}')
    accuracies += NBModel.score(features_test, target_test)
    precisions += precision_score(target_test, target_prediction)
    recalls += recall_score(target_test, target_prediction)
    f1_scores += f1_score(target_test, target_prediction)
    print("Under Sample Classification Report -->")
    print(classification_report(target_test, target_prediction))

print("Under Sampling Average Accuracy", (accuracies / n_splits) * 100)
print("Under Sampling Average Precision", (precisions / n_splits) * 100)
print("Under Sampling Average Recall", (recalls / n_splits) * 100)
print("Under Sampling Average F1 Score", (f1_scores / n_splits) * 100)