# Importing the dependencies
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef
from sklearn.metrics import plot_confusion_matrix
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

# Balancing the data by the under sampling method
# Building a sample dataset containing similar distribution of legit and fraud transactions
# Take random sample of the legit transactions
legit_sample = legit.sample(n=492)
print(legit_sample)

# Concatenating the legit_sample and the fraud dataset
new_dataset = pd.concat([legit_sample, fraud], axis=0)
new_dataset_smot_over = pd.concat([legit, fraud], axis=0)
print(new_dataset.head())
print(new_dataset.tail())
print(new_dataset['Class'].value_counts())

# mean fo the new dataset to check if we have a good sample
print(new_dataset.groupby('Class').mean())

# Splitting the dataset into Features & Target
features = new_dataset.drop(columns='Class', axis=1)
target = new_dataset['Class']
features_smot_over = new_dataset_smot_over.drop(columns='Class', axis=1)
target_smot_over = new_dataset_smot_over['Class']
print(features)
print(target)

# Splitting the data into Training data & Testing data
features_train, features_test, target_train, target_test = train_test_split(features, target,
                                                                            test_size=0.2, stratify=target,
                                                                            random_state=random_state)

features_train_smot_over, features_test_smot_over, target_train_smot_over, target_test_smot_over = train_test_split(
    features_smot_over, target_smot_over,
    test_size=0.2, stratify=target_smot_over,
    random_state=random_state)

print(features.shape, features_train.shape, features_test.shape)

# SMOT Technique
smote = SMOTE(random_state=random_state)
features_train_smot, target_train_smot = smote.fit_resample(features_train_smot_over, target_train_smot_over)

# Random Over Sampling Technique
overSample = RandomOverSampler(sampling_strategy='minority')
features_train_over, target_train_over = overSample.fit_resample(features_train_smot_over, target_train_smot_over)

# Create a decision tree and fit it to the training data
clf_dt = DecisionTreeClassifier(random_state=random_state)
# Model for under sampling technique
clf_dt = clf_dt.fit(features_train, target_train)
predicted_data = clf_dt.predict(features_test)

# Model for SMOT
clf_dt_smot = clf_dt.fit(features_train_smot, target_train_smot)
predicted_data_smot = clf_dt_smot.predict(features_test_smot_over)

# Model for Random Over Sampling
clf_dt_over = clf_dt.fit(features_train_over, target_train_over)
predicted_data_over = clf_dt_smot.predict(features_test_smot_over)

DT_accuracy_score = accuracy_score(target_test, predicted_data)
DT_precision_score = precision_score(target_test, predicted_data)
DT_recall_score = recall_score(target_test, predicted_data)
DT_f1_score = f1_score(target_test, predicted_data)
print("Accuracy before pruning for Under Sampling Technique -->", DT_accuracy_score)
print("Precision Under Sampling", DT_precision_score)
print("Recall Under Sampling", DT_recall_score)
print("F1 Score Under Sampling", DT_f1_score)


DT_accuracy_score_smot = accuracy_score(target_test_smot_over, predicted_data_smot)
DT_precision_score_smot = precision_score(target_test_smot_over, predicted_data_smot)
DT_recall_score_smot = recall_score(target_test_smot_over, predicted_data_smot)
DT_f1_score_smot = f1_score(target_test_smot_over, predicted_data_smot)

print("Accuracy before pruning for SMOT Technique -->", DT_accuracy_score_smot)
print("Precision SMOT", DT_precision_score_smot)
print("Recall SMOT", DT_recall_score_smot)
print("F1 Score SMOT", DT_f1_score_smot)

DT_accuracy_score_over = accuracy_score(target_test_smot_over, predicted_data_over)
DT_precision_score_over = precision_score(target_test_smot_over, predicted_data_over)
DT_recall_score_over = recall_score(target_test_smot_over, predicted_data_over)
DT_f1_score_over = f1_score(target_test_smot_over, predicted_data_over)

print("Accuracy before pruning for Over Sampling Technique -->", DT_accuracy_score_over)
print("Precision Over Sampling", DT_precision_score_over)
print("Recall Over Sampling", DT_recall_score_over)
print("F1 Score Over Sampling", DT_f1_score_over)

# Plotting the tree for Under Sampling technique
plt.figure(figsize=(15, 7.5))
plt.title("Tree for Under Sampling Technique")
plot_tree(clf_dt, filled=True, rounded=False, class_names=["Legit", "Fraud"],
          feature_names=features.columns)
plt.show()

# Plot Confusion Matrix for Under Sampling Technique
plt.figure(figsize=(12, 12))
plot_confusion_matrix(clf_dt, features_test, target_test, display_labels=["Normal", "Fraud"])
plt.title("Confusion matrix for Under Sampling Technique")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

# Plotting the tree for SMOT technique
plt.figure(figsize=(15, 7.5))
plt.title("Tree for SMOT Technique")
plot_tree(clf_dt_smot, filled=True, rounded=False, class_names=["Legit", "Fraud"],
          feature_names=features_smot_over.columns)
plt.show()

# Plot Confusion Matrix for SMOT Technique
plt.figure(figsize=(12, 12))
plot_confusion_matrix(clf_dt_smot, features_test_smot_over, target_test_smot_over, display_labels=["Normal", "Fraud"])
plt.title("Confusion matrix for SMOT Technique")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

# Plotting the tree for Random Over Sampling technique
plt.figure(figsize=(15, 7.5))
plt.title("Tree for Random Over Sampling Technique")
plot_tree(clf_dt, filled=True, rounded=False, class_names=["Legit", "Fraud"],
          feature_names=features_smot_over.columns)
plt.show()

# Plot Confusion Matrix for Random Over Sampling Technique
plt.figure(figsize=(12, 12))
plot_confusion_matrix(clf_dt_over, features_test_smot_over, target_test_smot_over, display_labels=["Normal", "Fraud"])
plt.title("Confusion matrix for Random Over Sampling Technique")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

# Decision tree Pruning
path = clf_dt.cost_complexity_pruning_path(features_train, target_train)  # determine the values of alpha
ccp_alphas, impurities = path.ccp_alphas, path.impurities
ccp_alphas = ccp_alphas[:-1]  # exclude the maximum value for alpha

path_smot = clf_dt_smot.cost_complexity_pruning_path(features_train_smot,
                                                     target_train_smot)  # determine the values of alpha
ccp_alphas_smot, impurities_smot = path_smot.ccp_alphas, path_smot.impurities
ccp_alphas_smot = ccp_alphas_smot[:-1]  # exclude the maximum value for alpha

path_over = clf_dt_over.cost_complexity_pruning_path(features_train_over,
                                                     target_train_over)  # determine the values of alpha
ccp_alphas_over, impurities_over = path_over.ccp_alphas, path_over.impurities
ccp_alphas_over = ccp_alphas_over[:-1]  # exclude the maximum value for alpha

clf_dts = []  # array to put decision trees for under sampling technique into
maximum = 0
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=random_state, ccp_alpha=ccp_alpha)
    clf_dt.fit(features_train, target_train)
    clf_dts.append(clf_dt)

clf_dts_smot = []  # array to put decision trees for SMOT technique into
maximum_smot = 0
for ccp_alphas_smot in ccp_alphas_smot:
    clf_dt_smot = DecisionTreeClassifier(random_state=random_state, ccp_alpha=ccp_alphas_smot)
    clf_dt_smot.fit(features_train_smot, target_train_smot)
    clf_dts_smot.append(clf_dt_smot)

clf_dts_over = []  # array to put decision trees for Random Over Sampling Technique into
maximum_over = 0
for ccp_alphas_over in ccp_alphas_over:
    clf_dt_over = DecisionTreeClassifier(random_state=random_state, ccp_alpha=ccp_alphas_over)
    clf_dt_over.fit(features_train_over, target_train_over)
    clf_dts_over.append(clf_dt_over)

# Graph the accuracy of the trees of the under sampling technique using Training dataset and Testing dataset as a function of alpha
train_scores = [clf_dt.score(features_train, target_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(features_test, target_test) for clf_dt in clf_dts]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets for under sampling technique")
ax.plot(ccp_alphas, train_scores, marker='o', label='train', drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label='test', drawstyle="steps-post")
ax.legend()
plt.show()

# After alpha is about 0.016 the accuracy of the Training dataset drops-off and that suggests that ccp_alpha=0.016
clf_dt = DecisionTreeClassifier(ccp_alpha=0.006)
scores = cross_val_score(clf_dt, features_train, target_train, cv=5)
df = pd.DataFrame(data={'tree': range(5), 'accuracy': scores})
df.plot(x='tree', y='accuracy', marker='o', linestyle='--')

# use cross validation to find the optimal value for ccp_value
# create an array to store the results of each fold during cross validation
alpha_loop_values = []

# for each candidate value for alpha, we will run 5-fold cross validations
# then we will store the mean and the standard deviation of the accuracy scores for each call
# to cross_val_score in alpha_loop_values
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=random_state, ccp_alpha=ccp_alpha)
    scores = cross_val_score(clf_dt, features_train, target_train, cv=5)
    alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])

# Now we can draw a graph of the mean and the standard deviations of the scores
# for each candidate value for alpha
alpha_results = pd.DataFrame(alpha_loop_values, columns=['alpha', 'mean_accuracy', 'std'])
alpha_results.plot(x='alpha', y='mean_accuracy', yerr='std', marker='o', linestyle='--')
plt.show()
print(alpha_results)

# Store the ideal value of alpha to build the best tree
ideal_ccp_alpha_index = alpha_results['mean_accuracy'].idxmax()
ideal_ccp_alpha = alpha_results.iloc[ideal_ccp_alpha_index]['alpha']
print(ideal_ccp_alpha)

# Build and train new decision tree of the under sampling technique using the ideal ccp_alpha
clf_dt_pruned = DecisionTreeClassifier(random_state=random_state, ccp_alpha=ideal_ccp_alpha)
clf_dt_pruned = clf_dt_pruned.fit(features_train, target_train)
predicted_data = clf_dt_pruned.predict(features_test)

# Plotting the tree of the under sampling technique
plt.figure(figsize=(15, 7.5))
plt.title("Tree for under Sampling Technique")
plot_tree(clf_dt_pruned, filled=True, rounded=True, class_names=["Fraud", 'Legit'],
          feature_names=features.columns)
plt.show()

# Plotting the confusion matrix of the under sampling technique
plt.figure(figsize=(15, 7.5))
plt.title("Confusion Matrix for under Sampling Technique")
plot_confusion_matrix(clf_dt_pruned, features_test, target_test, display_labels=["Fraud", 'Legit'])
plt.show()

# Graph the accuracy of the trees of the under sampling technique using Training dataset and Testing dataset as a function of alpha
train_scores = [clf_dt.score(features_train, target_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(features_test, target_test) for clf_dt in clf_dts]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets for under sampling technique")
ax.plot(ccp_alphas, train_scores, marker='o', label='train', drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label='test', drawstyle="steps-post")
ax.legend()
plt.show()

# After alpha is about 0.016 the accuracy of the Training dataset drops-off and that suggests that ccp_alpha=0.016
clf_dt = DecisionTreeClassifier(ccp_alpha=0.006)
scores = cross_val_score(clf_dt, features_train, target_train, cv=5)
df = pd.DataFrame(data={'tree': range(5), 'accuracy': scores})
df.plot(x='tree', y='accuracy', marker='o', linestyle='--')

# use cross validation to find the optimal value for ccp_value
# create an array to store the results of each fold during cross validation
alpha_loop_values = []

# for each candidate value for alpha, we will run 5-fold cross validations
# then we will store the mean and the standard deviation of the accuracy scores for each call
# to cross_val_score in alpha_loop_values
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=random_state, ccp_alpha=ccp_alpha)
    scores = cross_val_score(clf_dt, features_train, target_train, cv=5)
    alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])

# Now we can draw a graph of the mean and the standard deviations of the scores
# for each candidate value for alpha
alpha_results = pd.DataFrame(alpha_loop_values, columns=['alpha', 'mean_accuracy', 'std'])
alpha_results.plot(x='alpha', y='mean_accuracy', yerr='std', marker='o', linestyle='--')
plt.show()
print(alpha_results)

# Store the ideal value of alpha to build the best tree
ideal_ccp_alpha_index = alpha_results['mean_accuracy'].idxmax()
ideal_ccp_alpha = alpha_results.iloc[ideal_ccp_alpha_index]['alpha']
print(ideal_ccp_alpha)

# Build and train new decision tree of the under sampling technique using the ideal ccp_alpha
clf_dt_pruned = DecisionTreeClassifier(random_state=random_state, ccp_alpha=ideal_ccp_alpha)
clf_dt_pruned = clf_dt_pruned.fit(features_train, target_train)
predicted_data = clf_dt_pruned.predict(features_test)
accuracy_score += accuracy_score(target_test, predicted_data)
precision_score += precision_score(target_test, predicted_data)
recall_score += recall_score(target_test, predicted_data)
f1_score += f1_score(target_test, predicted_data)
MCC = matthews_corrcoef(target_test, predicted_data)

print("Mean Accuracy Under Sampling", accuracy_score)
print("Mean Precision Under Sampling", precision_score)
print("Mean Recall Under Sampling", recall_score)
print("Mean F1 Score Under Sampling", f1_score)
print("Mean MCC Under Sampling", MCC)

# Plotting the tree of the under sampling technique
plt.figure(figsize=(15, 7.5))
plt.title("Tree for under Sampling Technique")
plot_tree(clf_dt_pruned, filled=True, rounded=True, class_names=["Fraud", 'Legit'],
          feature_names=features.columns)
plt.show()

# Plotting the confusion matrix of the under sampling technique
plt.figure(figsize=(15, 7.5))
plt.title("Confusion Matrix for under Sampling Technique")
plot_confusion_matrix(clf_dt_pruned, features_test, target_test, display_labels=["Fraud", 'Legit'])
plt.show()

# Graph the accuracy of the trees of the SMOT technique using Training dataset and Testing dataset as a function of alpha
train_scores_smot = [clf_dt_smot.score(features_train_smot, target_train_smot) for clf_dt_smot in clf_dts_smot]
test_scores_smot = [clf_dt_smot.score(features_test_smot_over, target_test_smot_over) for clf_dt_smot in clf_dts_smot]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets for SMOT technique")
ax.plot(ccp_alphas_smot, train_scores_smot, marker='o', label='train', drawstyle="steps-post")
ax.plot(ccp_alphas_smot, test_scores_smot, marker='o', label='test', drawstyle="steps-post")
ax.legend()
plt.show()

# After alpha is about 0.016 the accuracy of the Training dataset drops-off and that suggests that ccp_alpha=0.016
clf_dt_smot = DecisionTreeClassifier(ccp_alpha=0.006)
scores_smot = cross_val_score(clf_dt_smot, features_train_smot, target_train_smot, cv=5)
df_smot = pd.DataFrame(data={'tree': range(5), 'accuracy': scores_smot})
df_smot.plot(x='tree', y='accuracy', marker='o', linestyle='--')

# use cross validation to find the optimal value for ccp_value
# create an array to store the results of each fold during cross validation
alpha_loop_values_smot = []

# for each candidate value for alpha, we will run 5-fold cross validations
# then we will store the mean and the standard deviation of the accuracy scores for each call
# to cross_val_score in alpha_loop_values
for ccp_alphas_smot in ccp_alphas_smot:
    clf_dt_smot = DecisionTreeClassifier(random_state=random_state, ccp_alpha=ccp_alphas_smot)
    scores_smot = cross_val_score(clf_dt_smot, features_train_smot, target_train_smot, cv=5)
    alpha_loop_values_smot.append([ccp_alphas_smot, np.mean(scores_smot), np.std(scores_smot)])

# Now we can draw a graph of the mean and the standard deviations of the scores
# for each candidate value for alpha
alpha_results_smot = pd.DataFrame(alpha_loop_values_smot, columns=['alpha', 'mean_accuracy', 'std'])
alpha_results_smot.plot(x='alpha', y='mean_accuracy', yerr='std', marker='o', linestyle='--')
plt.title("Alpha for SMOT Technique")
plt.show()
print(alpha_results_smot)

# Store the ideal value of alpha to build the best tree
ideal_ccp_alpha_index_smot = alpha_results_smot['mean_accuracy'].idxmax()
ideal_ccp_alpha_smot = alpha_results_smot.iloc[ideal_ccp_alpha_index_smot]['alpha']
print(ideal_ccp_alpha_smot)

# Build and train new decision tree of the under sampling technique using the ideal ccp_alpha
clf_dt_pruned_smot = DecisionTreeClassifier(random_state=random_state, ccp_alpha=ideal_ccp_alpha_smot)
clf_dt_pruned_smot = clf_dt_pruned_smot.fit(features_train_smot, target_train_smot)
predicted_data_smot = clf_dt_pruned_smot.predict(features_test_smot_over)
accuracy_score += accuracy_score(target_test, predicted_data_smot)
precision_score += precision_score(target_test, predicted_data_smot)
recall_score += recall_score(target_test, predicted_data_smot)
f1_score += f1_score(target_test, predicted_data_smot)
MCC = matthews_corrcoef(target_test, predicted_data_smot)

print("Mean Accuracy SMOT", accuracy_score)
print("Mean Precision SMOT", precision_score)
print("Mean Recall SMOT", recall_score)
print("Mean F1 Score SMOT", f1_score)
print("Mean MCC SMOT", MCC)

# Plotting the tree of the under sampling technique
plt.figure(figsize=(15, 7.5))
plt.title("Tree for SMOT Technique")
plot_tree(clf_dt_pruned_smot, filled=True, rounded=True, class_names=["Fraud", 'Legit'],
          feature_names=features_smot_over.columns)
plt.show()

# Plotting the confusion matrix of the under sampling technique
plt.figure(figsize=(15, 7.5))
plt.title("Confusion Matrix for SMOT Technique")
plot_confusion_matrix(clf_dt_pruned_smot, features_test_smot_over, target_test_smot_over,
                      display_labels=["Fraud", 'Legit'])
plt.show()

# Graph the accuracy of the trees of the Random Over sampling technique using Training dataset and Testing dataset as
# a function of alpha
train_scores_over = [clf_dt_over.score(features_train_over, target_train_over) for clf_dt_over in clf_dts_over]
test_scores_over = [clf_dt_over.score(features_test_smot_over, target_test_smot_over) for clf_dt_over in clf_dts_over]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets for Random Over sampling technique")
ax.plot(ccp_alphas_over, train_scores_over, marker='o', label='train', drawstyle="steps-post")
ax.plot(ccp_alphas_over, test_scores_over, marker='o', label='test', drawstyle="steps-post")
ax.legend()
plt.show()

# After alpha is about 0.016 the accuracy of the Training dataset drops-off and that suggests that ccp_alpha=0.016
clf_dt_over = DecisionTreeClassifier(ccp_alpha=0.006)
scores_over = cross_val_score(clf_dt_over, features_train_over, target_train_over, cv=5)
df_over = pd.DataFrame(data={'tree': range(5), 'accuracy': scores_over})
df.plot(x='tree', y='accuracy', marker='o', linestyle='--')

# use cross validation to find the optimal value for ccp_value
# create an array to store the results of each fold during cross validation
alpha_loop_values_over = []

# for each candidate value for alpha, we will run 5-fold cross validations
# then we will store the mean and the standard deviation of the accuracy scores for each call
# to cross_val_score in alpha_loop_values
for ccp_alphas_over in ccp_alphas_over:
    clf_dt_over = DecisionTreeClassifier(random_state=random_state, ccp_alpha=ccp_alphas_over)
    scores_over = cross_val_score(clf_dt_over, features_train_over, target_train_over, cv=5)
    alpha_loop_values_over.append([ccp_alphas_over, np.mean(scores_over), np.std(scores_over)])

# Now we can draw a graph of the mean and the standard deviations of the scores
# for each candidate value for alpha
alpha_results_over = pd.DataFrame(alpha_loop_values_over, columns=['alpha', 'mean_accuracy', 'std'])
alpha_results_over.plot(x='alpha', y='mean_accuracy', yerr='std', marker='o', linestyle='--')
plt.title("Alphas for Random Over Sampling Technique")
plt.show()
print(alpha_results_over)

# Store the ideal value of alpha to build the best tree
ideal_ccp_alpha_index_over = alpha_results_over['mean_accuracy'].idxmax()
ideal_ccp_alpha_over = alpha_results_over.iloc[ideal_ccp_alpha_index_over]['alpha']
print(ideal_ccp_alpha_over)

# Build and train new decision tree of the under sampling technique using the ideal ccp_alpha
clf_dt_pruned_over = DecisionTreeClassifier(random_state=random_state, ccp_alpha=ideal_ccp_alpha_over)
clf_dt_pruned_over = clf_dt_pruned_over.fit(features_train_over, target_train_over)
predicted_data_over = clf_dt_pruned_over.predict(features_test_smot_over)
accuracy_score += accuracy_score(target_test, predicted_data_over)
precision_score += precision_score(target_test, predicted_data_over)
recall_score += recall_score(target_test, predicted_data_over)
f1_score += f1_score(target_test, predicted_data_over)
MCC = matthews_corrcoef(target_test, predicted_data_over)

print("Mean Accuracy Over Sampling", accuracy_score)
print("Mean Precision Over Sampling", precision_score)
print("Mean Recall Over Sampling", recall_score)
print("Mean F1 Score Over Sampling", f1_score)
print("Mean MCC Over Sampling", MCC)

# Plotting the tree of the Random Over sampling technique
plt.figure(figsize=(15, 7.5))
plt.title("Tree for Random Over Sampling Technique")
plot_tree(clf_dt_pruned_over, filled=True, rounded=True, class_names=["Fraud", 'Legit'],
          feature_names=features_smot_over.columns)
plt.show()

# Plotting the confusion matrix of the under sampling technique
plt.figure(figsize=(15, 7.5))
plt.title("Confusion Matrix for Random Over Sampling Technique")
plot_confusion_matrix(clf_dt_pruned_over, features_test_smot_over, target_test_smot_over,
                      display_labels=["Fraud", 'Legit'])
plt.show()

# scoring Decision tress
DT_accuracy_score = accuracy_score(predicted_data, target_test)
DT_precision_score = precision_score(predicted_data, target_test)
DT_recall_score = recall_score(predicted_data, target_test)
DT_f1_score = f1_score(predicted_data, target_test)
DT_MCC = matthews_corrcoef(predicted_data, target_test)

# printing
print("")
print("Decision Trees")
print("Scores")
print("Accuracy -->", DT_accuracy_score)
print("Precision -->", DT_precision_score)
print("Recall -->", DT_recall_score)
print("F1 -->", DT_f1_score)
print("MCC -->", DT_MCC)
