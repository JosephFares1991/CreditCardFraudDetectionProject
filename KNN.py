# Importing the dependencies
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, \
    matthews_corrcoef, classification_report
from imblearn.over_sampling import SMOTE, RandomOverSampler
from numpy.random import seed
import random

seed(1)
random.seed(2)
random_state = 42

# Read the data in the CSV file using pandas
df = pd.read_csv('creditcard.csv')
df.head()
df.shape

# Checking for missing values
# In this data set, there are no missing values. So we don't need to handle missing values in the dataset.

df.isnull().sum()

All = df.shape[0]
fraud = df[df['Class'] == 1]
nonFraud = df[df['Class'] == 0]


# Lets shuffle the data before creating the subsamples
df = df.sample(frac=1)
frauds = df[df['Class'] == 1]
non_frauds = df[df['Class'] == 0][:492]

# Non Frauds for SMOT
non_frauds_smot = df[df['Class'] == 0]

# Dataframe using under sampling
new_df = pd.concat([non_frauds, frauds])

# Dataframe using SMOT
new_df_smot = pd.concat([non_frauds_smot, frauds])

# Shuffle dataframe rows
new_df = new_df.sample(frac=1, random_state=random_state)
new_df.head()

# prepare the data
features = new_df.drop(['Class'], axis=1)
features_smot = new_df_smot.drop(['Class'], axis=1)

labels = pd.DataFrame(new_df['Class'])
labels_smot = pd.DataFrame(new_df_smot['Class'])

feature_array = features.values
feature_array_smot = features_smot.values

label_array = labels.values
label_array_smot = labels_smot.values

# For the model building I am using K Nearest Neighbors. So we need find an optimal K to get the best out of it.
neighbours = np.arange(1, 25)
train_accuracy = np.empty(len(neighbours))
test_accuracy = np.empty(len(neighbours))

train_accuracy_smot = np.empty(len(neighbours))
test_accuracy_smot = np.empty(len(neighbours))

train_accuracy_over = np.empty(len(neighbours))
test_accuracy_over = np.empty(len(neighbours))

accuracies = 0
precisions = 0
n_splits = 5
cv = StratifiedKFold(n_splits=n_splits)

knn_accuracy_score = 0
knn_precision_score = 0
knn_recall_score = 0
knn_f1_score = 0
knn_MCC = 0


smote = SMOTE(random_state=random_state)
overSample = RandomOverSampler(sampling_strategy='minority')

# Cross Validation Under Sampling

for train_idx, test_idx in cv.split(features, labels):
    features_train, target_train = features.iloc[train_idx], labels.iloc[train_idx]
    features_test, target_test = features.iloc[test_idx], labels.iloc[test_idx]

    for i, k in enumerate(neighbours):
        # Set up a knn classifier with k neighbors
        knn = KNeighborsClassifier(n_neighbors=k, algorithm="kd_tree", n_jobs=-1)

        # Fit the model for under sampling technique
        knn.fit(features_train, target_train)

        # Compute accuracy on the training set
        train_accuracy[i] = knn.score(features_train, target_train)

        # Compute accuracy on the test set
        test_accuracy[i] = knn.score(features_train, target_train)

        target_prediction = knn.predict(features_test)
        print(f'Accuracy Under Sampling: {knn.score(features_test, target_test)}')
        print(f'Precision Under Sampling: {precision_score(target_test, target_prediction)}')
        print("Under Sampling  Classification Report -->")
        print(classification_report(target_test, target_prediction))

    # Generate plot for Random Under Sampling technique
    plt.title('KNN Varying number of neighbors for Random Under Sampling technique')
    plt.plot(neighbours, test_accuracy, label='Testing Accuracy')
    plt.plot(neighbours, train_accuracy, label='Training accuracy')
    plt.legend()
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    plt.show()

    idx = np.where(test_accuracy == max(test_accuracy))
    x = neighbours[idx]

    knn = KNeighborsClassifier(n_neighbors=x[0], algorithm="kd_tree", n_jobs=2)
    knn.fit(features_train, target_train)
    knn_predicted_test_labels = knn.predict(features_test)

    # scoring knn Under Sampling
    knn_accuracy_score += accuracy_score(target_test, knn_predicted_test_labels)
    knn_precision_score += precision_score(target_test, knn_predicted_test_labels)
    knn_recall_score += recall_score(target_test, knn_predicted_test_labels)
    knn_f1_score += f1_score(target_test, knn_predicted_test_labels)
    knn_MCC += matthews_corrcoef(target_test, knn_predicted_test_labels)

print("Mean Accuracy Under Sampling", knn_accuracy_score / n_splits)
print("Mean Precision Under Sampling", knn_precision_score / n_splits)
print("Mean Recall Under Sampling", knn_recall_score / n_splits)
print("Mean F1 Score Under Sampling", knn_f1_score / n_splits)
print("Mean MCC Under Sampling", knn_MCC / n_splits)

# Cross Validation Over Sampling

for train_idx, test_idx in cv.split(features_smot, labels_smot):
    features_train, target_train = features_smot.iloc[train_idx], labels_smot.iloc[train_idx]
    features_test, target_test = features_smot.iloc[test_idx], labels_smot.iloc[test_idx]
    features_train, target_train = overSample.fit_resample(features_train, target_train)

    for i, k in enumerate(neighbours):
        # Set up a knn classifier with k neighbors
        knn = KNeighborsClassifier(n_neighbors=k, algorithm="kd_tree", n_jobs=-1)

        # Fit the model for under sampling technique
        knn.fit(features_train, target_train)

        # Compute accuracy on the training set
        train_accuracy_over[i] = knn.score(features_train, target_train)

        # Compute accuracy on the test set
        test_accuracy_over[i] = knn.score(features_train, target_train)

        target_prediction = knn.predict(features_test)
        print(f'Accuracy Over Sampling: {knn.score(features_test, target_test)}')
        print(f'Precision Over Sampling: {precision_score(target_test, target_prediction)}')
        print("Over Sampling  Classification Report -->")
        print(classification_report(target_test, target_prediction))

    # Generate plot for Random Under Sampling technique
    plt.title('KNN Varying number of neighbors for Random Over Sampling technique')
    plt.plot(neighbours, test_accuracy_over, label='Testing Accuracy')
    plt.plot(neighbours, train_accuracy_over, label='Training accuracy')
    plt.legend()
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    plt.show()

    idx = np.where(test_accuracy_over == max(test_accuracy_over))
    x = neighbours[idx]

    knn = KNeighborsClassifier(n_neighbors=x[0], algorithm="kd_tree", n_jobs=2)
    knn.fit(features_train, target_train)
    knn_predicted_test_labels = knn.predict(features_test)

    # scoring knn Over Sampling
    knn_accuracy_score += accuracy_score(target_test, knn_predicted_test_labels)
    knn_precision_score += precision_score(target_test, knn_predicted_test_labels)
    knn_recall_score += recall_score(target_test, knn_predicted_test_labels)
    knn_f1_score += f1_score(target_test, knn_predicted_test_labels)
    knn_MCC += matthews_corrcoef(target_test, knn_predicted_test_labels)

print("Mean Accuracy Over Sampling", knn_accuracy_score / n_splits)
print("Mean Precision Over Sampling", knn_precision_score / n_splits)
print("Mean Recall Over Sampling", knn_recall_score / n_splits)
print("Mean F1 Score Over Sampling", knn_f1_score / n_splits)
print("Mean MCC Over Sampling", knn_MCC / n_splits)

# Cross Validation SMOT

for train_idx, test_idx in cv.split(features_smot, labels_smot):
    features_train, target_train = features_smot.iloc[train_idx], labels_smot.iloc[train_idx]
    features_test, target_test = features_smot.iloc[test_idx], labels_smot.iloc[test_idx]
    features_train, target_train = smote.fit_resample(features_train, target_train)

    for i, k in enumerate(neighbours):
        # Set up a knn classifier with k neighbors
        knn = KNeighborsClassifier(n_neighbors=k, algorithm="kd_tree", n_jobs=-1)

        # Fit the model for under sampling technique
        knn.fit(features_train, target_train)

        # Compute accuracy on the training set
        train_accuracy_smot[i] = knn.score(features_train, target_train)

        # Compute accuracy on the test set
        test_accuracy_smot[i] = knn.score(features_train, target_train)

        target_prediction = knn.predict(features_test)
        print(f'Accuracy SMOT: {knn.score(features_test, target_test)}')
        print(f'Precision SMOT: {precision_score(target_test, target_prediction)}')
        print("SMOT  Classification Report -->")
        print(classification_report(target_test, target_prediction))

    # Generate plot for Random Under Sampling technique
    plt.title('KNN Varying number of neighbors for SMOT technique')
    plt.plot(neighbours, test_accuracy_smot, label='Testing Accuracy')
    plt.plot(neighbours, train_accuracy_smot, label='Training accuracy')
    plt.legend()
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    plt.show()

    idx = np.where(test_accuracy_smot == max(test_accuracy_smot))
    x = neighbours[idx]
    print('X', x)

    knn = KNeighborsClassifier(n_neighbors=x[0], algorithm="kd_tree", n_jobs=-1)
    knn.fit(features_train, target_train)
    knn_predicted_test_labels = knn.predict(features_test)

    # scoring knn SMOT Sampling
    knn_accuracy_score += accuracy_score(target_test, knn_predicted_test_labels)
    knn_precision_score += precision_score(target_test, knn_predicted_test_labels)
    knn_recall_score += recall_score(target_test, knn_predicted_test_labels)
    knn_f1_score += f1_score(target_test, knn_predicted_test_labels)
    knn_MCC += matthews_corrcoef(target_test, knn_predicted_test_labels)


print("Mean Accuracy SMOT", knn_accuracy_score/n_splits)
print("Mean Precision SMOT", knn_precision_score/n_splits)
print("Mean Recall SMOT", knn_recall_score/n_splits)
print("Mean F1 Score SMOT", knn_f1_score/n_splits)
print("Mean MCC SMOT", knn_MCC/n_splits)
