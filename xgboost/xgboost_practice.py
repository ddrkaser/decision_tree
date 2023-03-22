import pandas as pd  # load and manipulate data and for One-Hot Encoding
import numpy as np  # calculate the mean and standard deviation
import xgboost as xgb  # XGBoost stuff
# split  data into training and testing sets
from sklearn.model_selection import train_test_split
# for scoring during cross validation
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV  # cross validation
from sklearn.metrics import confusion_matrix  # creates a confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
import multiprocessing
import graphviz

# Now we load in a dataset from the IBM Base Samples. Specifically, we are going to use the Telco Churn Dataset. This dataset will allow us to predict if someone will stop using Telco's services or not using a variety of continuous and categorical datatypes.
df = pd.read_csv('Telco_customer_churn.csv')
df.head()
df.columns

df.drop(['Churn Label', 'Churn Score', 'CLTV', 'Churn Reason'], axis=1,
        inplace=True)  # set axis=0 to remove rows, axis=1 to remove columns
df.columns
df['Count'].unique()
df['Country'].unique()
df['State'].unique()
df['City'].unique()

df.drop(['CustomerID', 'Count', 'Country', 'State', 'Lat Long'], axis=1,
        inplace=True)  # set axis=0 to remove rows, axis=1 to remove columns


"""Although it is OK to have whitespace in the city names in City for XGBoost and classification, we can't have any whitespace if we want to draw a tree. So let's take care of that now by replacing the white space in the city names with an underscore character _."""
df['City'].replace(' ', '_', regex=True, inplace=True)

df.columns = df.columns.str.replace(' ', '_')
df.columns

# missing data
df.dtypes
df['Phone_Service'].unique()
df['Total_Charges'].unique()
dd = df['Total_Charges'].value_counts()
dd.index[1]

#df['Total Charges'] = pd.to_numeric(df['Total_Charges'])
len(df.loc[df['Total_Charges'] == ' '])
df.loc[(df['Total_Charges'] == ' '), 'Total_Charges'] = 0
df.loc[df['Tenure_Months'] == 0]
df['Total_Charges'] = df['Total_Charges'].astype('float')

df.replace(' ', '_', regex=True, inplace=True)
df.size

# split data
X = df.drop('Churn_Value', axis=1).copy()
y = df['Churn_Value'].copy()

# One-Hot Encoding, let's just see what happens when we convert Payment_Method without saving the results.
pd.get_dummies(X, columns=['Payment_Method']).columns

X_encoded = pd.get_dummies(X, columns=['City',
                                       'Gender',
                                       'Senior_Citizen',
                                       'Partner',
                                       'Dependents',
                                       'Phone_Service',
                                       'Multiple_Lines',
                                       'Internet_Service',
                                       'Online_Security',
                                       'Online_Backup',
                                       'Device_Protection',
                                       'Tech_Support',
                                       'Streaming_TV',
                                       'Streaming_Movies',
                                       'Contract',
                                       'Paperless_Billing',
                                       'Payment_Method'])
X_encoded.head()
y.unique()

# This data is imbalanced by dividing the number of people who left the company, where y = 1, by the total number of people in the dataset.
sum(y)/len(y)
# split the dataset using stratification in order to maintain the same percentage of people who left the company in both the training set and the testing set.
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42, stratify=y)
# verify
sum(y_train)/len(y_train)
sum(y_test)/len(y_test)

# clf_xgb = xgb.XGBClassifier(objective='binary:logistic',
#                             eval_metric="logloss", ## this avoids a warning...
#                             seed=42,
#                             use_label_encoder=False)

clf_xgb = xgb.XGBClassifier(objective='binary:logistic', ## this avoids a warning...
                            seed=42)
## NOTE: newer versions of XGBoost will issue a warning if you don't explitly tell it that
## you are not expecting it to do label encoding on its own (in other words, since we
## have ensured that the categorical values are all numeric, we do not expect XGBoost to do label encoding),
## so we set use_label_encoder=False
clf_xgb.fit(X_train,
            y_train,
            verbose=True,
            ## the next three arguments set up early stopping.
            early_stopping_rounds=10,
            eval_metric='aucpr',
            eval_set=[(X_test, y_test)])

# plot_confusion_matrix(clf_xgb,
#                       X_test,
#                       y_test,
#                       values_format='d',
#                       display_labels=["Did not leave", "Left"])

ConfusionMatrixDisplay.from_estimator(clf_xgb, X_test, y_test,values_format='d', display_labels=["Did not leave", "Left"])

#optimization

# ROUND 1
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.01, 0.05],
    'gamma': [0, 0.25, 1.0],
    'reg_lambda': [0, 1.0, 10.0],
    'scale_pos_weight': [1, 3, 5] # NOTE: XGBoost recommends sum(negative instances) / sum(positive instances)
}
# Output: max_depth: 4, learning_rate: 0.1, gamma: 0.25, reg_lambda: 10, scale_pos_weight: 3
# Because learning_rate and reg_lambda were at the ends of their range, we will continue to explore those...


# NOTE: To speed up cross validiation, and to further prevent overfitting.
# We are only using a random subset of the data (90%) and are only
# using a random subset of the features (columns) (50%) per tree.
optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic',
                                seed=42,
                                subsample=0.9,
                                colsample_bytree=0.5),
    param_grid=param_grid,
    scoring='roc_auc', ## see https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    verbose=0, # NOTE: If you want to see what Grid Search is doing, set verbose=2
    n_jobs =1,
    cv = 3
)

optimal_params.fit(X_train,
                    y_train,
                    early_stopping_rounds=10,
                    eval_metric='auc',
                    eval_set=[(X_test, y_test)],
                    verbose=False)


print(optimal_params.best_params_)
#{'gamma': 0.25, 'learning_rate': 0.1, 'max_depth': 4, 'reg_lambda': 10.0, 'scale_pos_weight': 1}

## ROUND 2
param_grid = {
    'max_depth': [4],
    'learning_rate': [0.1, 0.5, 1],
    'gamma': [0.25],
    'reg_lambda': [10.0, 20, 100],
      'scale_pos_weight': [1,3]
}

optimal_params.fit(X_train,
                    y_train,
                    early_stopping_rounds=10,
                    eval_metric='auc',
                    eval_set=[(X_test, y_test)],
                    verbose=False)
print(optimal_params.best_params_)

#final model
clf_xgb = xgb.XGBClassifier(seed=42,
                        objective='binary:logistic',
                        eval_metric='aucpr',
                        early_stopping_rounds=10,
                        gamma=0.25,
                        learning_rate=0.1,
                        max_depth=4,
                        reg_lambda=10,
                        scale_pos_weight=3,
                        subsample=0.9,
                        colsample_bytree=0.5)


clf_xgb.fit(X_train,
            y_train,
            verbose=True,
            eval_set=[(X_test, y_test)])

ConfusionMatrixDisplay.from_estimator(clf_xgb, X_test, y_test,values_format='d', display_labels=["Did not leave", "Left"])


clf_xgb = xgb.XGBClassifier(seed=42,
                        eval_metric='aucpr', ## this avoids another warning...
                        objective='binary:logistic',
                        gamma=0.25,
                        learning_rate=0.1,
                        max_depth=4,
                        reg_lambda=10,
                        scale_pos_weight=3,
                        subsample=0.9,
                        colsample_bytree=0.5,
                        n_estimators=1 ## We set this to 1 so we can get gain, cover etc.)
                        )

clf_xgb.fit(X_train, y_train)

## now print out the weight, gain, cover etc. for the tree
## weight = number of times a feature is used in a branch or root across all trees
## gain = the average gain across all splits that the feature is used in
## cover = the average coverage across all splits a feature is used in
## total_gain = the total gain across all splits the feature is used in
## total_cover = the total coverage across all splits the feature is used in
## NOTE: Since we only built one tree, gain = total_gain and cover=total_cover
bst = clf_xgb.get_booster()
for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):
    print('%s: ' % importance_type, bst.get_score(importance_type=importance_type))

node_params = {'shape': 'box', ## make the nodes fancy
               'style': 'filled, rounded',
               'fillcolor': '#78cbe'}
leaf_params = {'shape': 'box',
               'style': 'filled',
               'fillcolor': '#e48038'}

xgb.to_graphviz(clf_xgb, num_trees=0,
                condition_node_params=node_params,
                leaf_node_params=leaf_params)