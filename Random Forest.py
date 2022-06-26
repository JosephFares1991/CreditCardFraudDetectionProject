# Importing the dependencies
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix
from imblearn.pipeline import Pipeline
from itertools import product
from numpy.random import seed
import random

seed(1)
random.seed(2)
random_state = 42

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

# SMOTE Technique
smote = SMOTE(random_state=random_state)
features_train_smot, target_train_smot = smote.fit_resample(features_train_smot_over, target_train_smot_over)

# Random Over Sampling Technique
overSample = RandomOverSampler(sampling_strategy='minority')
features_train_over, target_train_over = overSample.fit_resample(features_train_smot_over, target_train_smot_over)

# Feature Scaling
sc_features = StandardScaler()
features_train = sc_features.fit_transform(features_train)
features_train_smot = sc_features.fit_transform(features_train_smot)
features_train_over = sc_features.fit_transform(features_train_over)

features_test = sc_features.fit_transform(features_test)
features_test_smot = sc_features.fit_transform(features_test_smot_over)
features_test_over = sc_features.fit_transform(features_test_smot_over)

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=random_state)
rf.fit(features_train, target_train)
# Predicting the test set results
target_predictions = rf.predict(features_test)
# Accuracy on training set
print("Training Accuracy under sampling is: ", rf.score(features_train, target_train))
# Accuracy on testing set
print("Testing Accuracy under sampling is: ", rf.score(features_test, target_test))

rf.fit(features_train_smot, target_train_smot)
# Predicting the test set results
target_predictions_smot = rf.predict(features_test_smot)
# Accuracy on training set
print("Training Accuracy SMOT is: ", rf.score(features_train_smot, target_train_smot))
# Accuracy on testing set
print("Testing Accuracy smot is: ", rf.score(features_test_smot, target_test_smot_over))

rf.fit(features_train_over, target_train_over)
# Predicting the test set results
target_predictions_over = rf.predict(features_test_over)
# Accuracy on training set
print("Training Accuracy Random Over Sampling is: ", rf.score(features_train_over, target_train_over))
# Accuracy on testing set
print("Testing Accuracy Random Over Sampling is: ", rf.score(features_test_over, target_test_smot_over))

# Plotting the confusion matrix Under Sampling
plt.figure(figsize=(15, 7.5))
plt.title("Confusion Matrix for under Sampling Technique")
plot_confusion_matrix(rf, features_test, target_test, display_labels=["Fraud", 'Legit'])
plt.show()

# Plotting the confusion matrix SMOT
plt.figure(figsize=(15, 7.5))
plt.title("Confusion Matrix for SMOT Technique")
plot_confusion_matrix(rf, features_test_smot, target_test_smot_over, display_labels=["Fraud", 'Legit'])
plt.show()

# Plotting the confusion matrix SMOT
plt.figure(figsize=(15, 7.5))
plt.title("Confusion Matrix for Over Sampling Technique")
plot_confusion_matrix(rf, features_test_over, target_test_smot_over, display_labels=["Fraud", 'Legit'])
plt.show()


# Tuning Random Forest
kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
n_estimators = 100
max_features = [1, 'sqrt', 'log2']
max_depth = [None, 2, 3, 4, 5]
for f, d in product(max_features, max_depth):
    rf = RandomForestClassifier(n_estimators=n_estimators, criterion='entropy',
                                max_features=f, max_depth=d, n_jobs=2, random_state=random_state)
    cv_score_lr_mean_under = cross_val_score(rf, features, target, scoring="accuracy", cv=kf).mean() * 100
    cv_f1_lr_mean_under = cross_val_score(rf, features, target, scoring="f1", cv=kf).mean() * 100
    cv_precision_lr_mean_under = cross_val_score(rf, features, target, scoring="precision", cv=kf).mean() * 100
    cv_recall_lr_mean_under = cross_val_score(rf, features, target, scoring="recall", cv=kf).mean() * 100
    print('Classification Accuracy for Under Sampling on test sets with max_features = {} and max_depth = {} : {:.3f}'.
          format(f, d, cv_score_lr_mean_under))
    print('Classification F1 for Under Sampling on test sets with max_features = {} and max_depth = {} : {:.3f}'.
          format(f, d, cv_f1_lr_mean_under))
    print('Classification Precision for Under Sampling on test sets with max_features = {} and max_depth = {} : {:.3f}'.
          format(f, d, cv_precision_lr_mean_under))
    print('Classification Recall for Under Sampling on test sets with max_features = {} and max_depth = {} : {:.3f}'.
          format(f, d, cv_recall_lr_mean_under))
    print("-----------------------------------------------------------------------------------------------")
    # define pipeline
    steps = [('over', SMOTE()), ('model', rf)]
    pipeline = Pipeline(steps=steps)
    # evaluate pipeline
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=random_state)
    cv_score_lr_mean_smot = cross_val_score(pipeline, features_smot, target_smot, scoring="accuracy", cv=kf, n_jobs=-1).mean() * 100
    cv_f1_lr_mean_smot = cross_val_score(pipeline, features_smot, target_smot, scoring="f1", cv=kf, n_jobs=-1).mean() * 100
    cv_precision_lr_mean_smot = cross_val_score(pipeline, features_smot, target_smot, scoring="precision", cv=kf, n_jobs=-1).mean() * 100
    cv_recall_lr_mean_smot = cross_val_score(pipeline, features_smot, target_smot, scoring="recall", cv=kf, n_jobs=-1).mean() * 100
    print('Classification Accuracy for SMOT on test sets with max_features = {} and max_depth = {} : {:.3f}'.
          format(f, d, cv_score_lr_mean_smot))
    print('Classification F1 for SMOT on test sets with max_features = {} and max_depth = {} : {:.3f}'.
          format(f, d, cv_f1_lr_mean_smot))
    print('Classification Precision for SMOT on test sets with max_features = {} and max_depth = {} : {:.3f}'.
          format(f, d, cv_precision_lr_mean_smot))
    print('Classification Recall for SMOT on test sets with max_features = {} and max_depth = {} : {:.3f}'.
          format(f, d, cv_recall_lr_mean_smot))
    print("-----------------------------------------------------------------------------------------------")
    # define pipeline
    steps = [('over', RandomOverSampler()), ('model', rf)]
    pipeline = Pipeline(steps=steps)
    # evaluate pipeline
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=random_state)
    cv_score_lr_mean_over = cross_val_score(pipeline, features_smot, target_smot, scoring="accuracy", cv=kf,
                                            n_jobs=-1).mean() * 100
    cv_f1_lr_mean_over = cross_val_score(pipeline, features_smot, target_smot, scoring="f1", cv=kf,
                                         n_jobs=-1).mean() * 100
    cv_precision_lr_mean_over = cross_val_score(pipeline, features_smot, target_smot, scoring="precision", cv=kf,
                                                n_jobs=-1).mean() * 100
    cv_recall_lr_mean_over = cross_val_score(pipeline, features_smot, target_smot, scoring="recall", cv=kf,
                                             n_jobs=-1).mean() * 100
    print('Classification Accuracy for Random Over Sampling on test sets with max_features = {} and max_depth = {} : '
          '{:.3f}'.
          format(f, d, cv_score_lr_mean_over))
    print('Classification F1 for Random Over Sampling on test sets with max_features = {} and max_depth = {} : {:.3f}'.
          format(f, d, cv_f1_lr_mean_over))
    print('Classification Precision for Random Over Sampling on test sets with max_features = {} and max_depth = {} : '
          '{:.3f}'.
          format(f, d, cv_precision_lr_mean_over))
    print('Classification Recall for Random Over Sampling on test sets with max_features = {} and max_depth = {} : {'
          ':.3f}'.
          format(f, d, cv_recall_lr_mean_over))

