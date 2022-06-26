# Importing the dependencies
from sklearn.preprocessing import StandardScaler
from numpy.random import seed
import random


seed(1)
random.seed(2)
random_state = 42

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.model_selection import StratifiedKFold

cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Loading the dataset into pandas
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

# Balancing the data by the under sampling method
# Building a sample dataset containing similar distribution of legit and fraud transactions
# Take random sample of the legit transactions
legit_sample = legit.sample(n=492)
print(legit_sample)

# Concatenating the legit_sample and the fraud dataset
new_dataset = pd.concat([legit_sample, fraud], axis=0)
print(new_dataset.head())
print(new_dataset.tail())
print(new_dataset['Class'].value_counts())

# mean of the new dataset to check if we have a good sample
print(new_dataset.groupby('Class').mean())

# Splitting the dataset into Features & Target
features = new_dataset.drop(columns='Class', axis=1)
target = new_dataset['Class']

print(features)
print(target)

# Feature Scaling
sc_features = StandardScaler()

# Splitting the data into Training data & Testing data
features_train, features_test, target_train, target_test = train_test_split(features, target,
                                                                            test_size=0.2, stratify=target,
                                                                            random_state=random_state)
# Scaling Features
features_train = sc_features.fit_transform(features_train)
features_test = sc_features.fit_transform(features_test)

# Fitting SMOTE to the training set
# SMOT Dataset
new_dataset_smot = pd.concat([legit, fraud], axis=0)
features_smot = new_dataset_smot.drop(columns='Class', axis=1)
target_smot = new_dataset_smot['Class']

# Splitting the data into Training data & Testing data
features_train_smot_over, features_test_smot_over, target_train_smot_over, target_test_smot_over = train_test_split(
    features_smot,
    target_smot,
    test_size=0.2,
    stratify=target_smot,
    random_state=random_state)

smote = SMOTE(random_state=random_state)
features_train_smot, target_train_smot = smote.fit_resample(features_train_smot_over, target_train_smot_over)
features_train_smot = sc_features.fit_transform(features_train_smot)
features_test_smot = sc_features.fit_transform(features_test_smot_over)

# Random Over Sampling Technique
overSample = RandomOverSampler(sampling_strategy='minority')
features_train_over, target_train_over = overSample.fit_resample(features_train_smot_over, target_train_smot_over)
features_train_over = sc_features.fit_transform(features_train_over)
features_test_over = sc_features.fit_transform(features_test_smot_over)

# Logistic Regression model
model = LogisticRegression()

# Cross validation for under sampling
accuracies = 0
precisions = 0
recalls = 0
f1_scores = 0
n_splits = 5
cv = StratifiedKFold(n_splits=n_splits)
for train_idx, test_idx in cv.split(features, target):
    features_train, target_train = features.iloc[train_idx], target.iloc[train_idx]
    features_test, target_test = features.iloc[test_idx], target.iloc[test_idx]
    model.fit(features_train, target_train)
    target_prediction = model.predict(features_test)
    print(f'Accuracy: {model.score(features_test, target_test)}')
    print(f'Precision: {precision_score(target_test, target_prediction)}')
    accuracies += model.score(features_test, target_test)
    precisions += precision_score(target_test, target_prediction)
    recalls += recall_score(target_test, target_prediction)
    f1_scores += f1_score(target_test, target_prediction)
    print("Under Sampling Classification Report -->")
    print(classification_report(target_test, target_prediction))

print("Under Sampling Average Accuracy", (accuracies / n_splits) * 100)
print("Under Sampling Precision", (precisions / n_splits) * 100)
print("Under Sampling Average Recall", (recalls / n_splits) * 100)
print("Under Sampling Average F1 Score", (f1_scores / n_splits) * 100)

# Cross validation for SMOTE
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
    model.fit(features_train, target_train)
    target_prediction = model.predict(features_test)
    print(f'Accuracy: {model.score(features_test, target_test)}')
    print(f'Precision: {precision_score(target_test, target_prediction)}')
    accuracies += model.score(features_test, target_test)
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
n_splits = 5
cv = StratifiedKFold(n_splits=n_splits)
for train_idx, test_idx in cv.split(features_smot, target_smot):
    features_train, target_train = features_smot.iloc[train_idx], target_smot.iloc[train_idx]
    features_test, target_test = features_smot.iloc[test_idx], target_smot.iloc[test_idx]
    features_train, target_train = overSample.fit_resample(features_train, target_train)
    model.fit(features_train, target_train)
    target_prediction = model.predict(features_test)
    print(f'Accuracy: {model.score(features_test, target_test)}')
    print(f'Precision: {precision_score(target_test, target_prediction)}')
    accuracies += model.score(features_test, target_test)
    precisions += precision_score(target_test, target_prediction)
    recalls += recall_score(target_test, target_prediction)
    f1_scores += f1_score(target_test, target_prediction)
    print("Over Sample Classification Report -->")
    print(classification_report(target_test, target_prediction))

print("Random Over Sampling Average Accuracy", (accuracies / n_splits) * 100)
print("Random Over Sampling Average Precision", (precisions / n_splits) * 100)
print("Random Over Sampling Average Recall", (recalls / n_splits) * 100)
print("Random Over Sampling Average F1 Score", (f1_scores / n_splits) * 100)