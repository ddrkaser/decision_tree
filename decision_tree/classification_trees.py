import pandas as pd  # to load and manipulate data and for One-Hot Encoding
import numpy as np  # to calculate the mean and standard deviation
import matplotlib.pyplot as plt  # to draw graphs
from sklearn.tree import DecisionTreeClassifier  # to build a classification tree
from sklearn.tree import plot_tree  # to draw a classification tree
# to split data into training and testing sets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score  # for cross validation
from sklearn.metrics import confusion_matrix  # to create a confusion matrix
from sklearn.metrics import plot_confusion_matrix  # to draw a confusion matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import ConfusionMatrixDisplay

df = pd.read_csv('processed.cleveland.data', header=None)
df.head()
df.columns = ['age',
              'sex',
              'cp',
              'restbp',
              'chol',
              'fbs',
              'restecg',
              'thalach',
              'exang',
              'oldpeak',
              'slope',
              'ca',
              'thal',
              'hd']

df.dtypes
df.info()
df.describe()
df['ca'].unique()
df['thal'].unique()

len(df.loc[(df['ca'] == '?') | (df['thal'] == '?')])
df.loc[(df['ca'] == '?') | (df['thal'] == '?')]

#remove rows with missing values ?
df_no_NA = df.loc[(df['ca'] != '?') & (df['thal'] != '?')]
df_no_NA = df_no_NA.astype({'ca': 'float64','thal':'float64'})
df_no_NA.dtypes

#split dataset to train_X, label_y
X = df_no_NA.drop('hd', axis = 1).copy()
y = df_no_NA['hd'].copy()

#encoding categarical data
X.dtypes

#use get_dummies from pandas
pd.get_dummies(X, columns=['cp']).head()
X_encoded = pd.get_dummies(X, columns=['cp','restecg', 'slope','thal'])


#Now, one last thing before we build a Classification Tree. y doesn't just contain 0s and 1s. Instead, it has 5 different levels of heart disease. 0 = no heart disease and 1-4 are various degrees of heart disease.
#we're only making a tree that does simple classification and only care if someone has heart disease or not,
y.unique()
y[y > 0] = 1

#split training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42)

## create a decisiont tree and fit it to the training data
clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt = clf_dt.fit(X_train, y_train)

clf_dt.score(X_test, y_test)

plt.figure(figsize=(60, 30))
plot_tree(clf_dt,
          filled=True,
          rounded=True,
          class_names=["No HD", "Yes HD"],
          feature_names=X_encoded.columns);

plot_confusion_matrix(clf_dt, X_test, y_test, display_labels=["Does not have HD", "Has HD"])
ConfusionMatrixDisplay.from_estimator(clf_dt, X_test, y_test, display_labels=["Does not have HD", "Has HD"])

#cost complexity pruning
path = clf_dt.cost_complexity_pruning_path(X_train, y_train) # determine values for alpha
ccp_alphas = path.ccp_alphas # extract different values for alpha
ccp_alphas = ccp_alphas[:-1] # exclude the maximum value for alpha

clf_dts = [] # create an array that we will put decision trees into

## now create one decision tree per value for alpha and store it in the array
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf_dt.fit(X_train, y_train)
    clf_dts.append(clf_dt)

train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test", drawstyle="steps-post")
ax.legend()
plt.show()

#cross validation
clf_dt = DecisionTreeClassifier(random_state=42, ccp_alpha=0.016) # create the tree with ccp_alpha=0.016

## now use 5-fold cross validation create 5 different training and testing datasets that
## are then used to train and test the tree.
## NOTE: We use 5-fold because we don't have tons of data...
scores = cross_val_score(clf_dt, X_train, y_train, cv=5)
df = pd.DataFrame(data={'tree': range(5), 'accuracy': scores})

df.plot(x='tree', y='accuracy', marker='o', linestyle='--')
"""The graph above shows that using different Training and Testing data with the same alpha resulted in different accuracies, suggesting that alpha is sensitive to the datasets. So, instead of picking a single Training dataset and single Testing dataset, let's use cross validation to find the optimal value for ccp_alpha."""

## create an array to store the results of each fold during cross validiation
alpha_loop_values = []

## For each candidate value for alpha, we will run 5-fold cross validation.
## Then we will store the mean and standard deviation of the scores (the accuracy) for each call
## to cross_val_score in alpha_loop_values...
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    scores = cross_val_score(clf_dt, X_train, y_train, cv=5)
    alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])

## Now we can draw a graph of the means and standard deviations of the scores
## for each candidate value for alpha
alpha_results = pd.DataFrame(alpha_loop_values,
                             columns=['alpha', 'mean_accuracy', 'std'])

alpha_results.plot(x='alpha',
                   y='mean_accuracy',
                   yerr='std',
                   marker='o',
                   linestyle='--')

ideal_ccp_alpha = float(alpha_results[(alpha_results['alpha'] > 0.014)&(alpha_results['alpha'] < 0.015)]['alpha'])

## Build and train a new decision tree, only this time use the optimal value for alpha
clf_dt_pruned = DecisionTreeClassifier(random_state=42, ccp_alpha=ideal_ccp_alpha)
clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)
clf_dt_pruned.score(X_test, y_test)
ConfusionMatrixDisplay.from_estimator(clf_dt_pruned, X_test, y_test, display_labels=["Does not have HD", "Has HD"])

plt.figure(figsize=(60, 30))
plot_tree(clf_dt_pruned,
          filled=True,
          rounded=True,
          class_names=["No HD", "Yes HD"],
          feature_names=X_encoded.columns);
