"""
https://www.kaggle.com/code/robikscube/time-series-forecasting-with-machine-learning-yt
https://www.kaggle.com/code/robikscube/pt2-time-series-forecasting-with-xgboost/notebook
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV  # cross validation

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

df = pd.read_csv('PJME_hourly.csv')
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)
df = df.sort_index()
df.dtypes

df.plot(style='.',
        figsize=(15, 5),
        color=color_pal[0],
        title='PJME Energy Use in MW')

# Outlier Analysis and removal
#the hist
df['PJME_MW'].plot(kind='hist', bins=500)
#most data >20000
df[df['PJME_MW'] < 20000]['PJME_MW'].plot(style='.',
                                             figsize=(15, 5),
                                             color=color_pal[5],
                                             title='Outliers')

#these could be some extrem cases or sensors are not working properly
df[df['PJME_MW'] < 19000]['PJME_MW'].plot(style='.',
                                             figsize=(15, 5),
                                             color=color_pal[5],
                                             title='Outliers')
#remove outliers
df = df[df['PJME_MW'] > 19000]

# test/train split
train = df.iloc[df.index < '01-01-2015']
test = df.iloc[df.index >= '01-01-2015']

# visulize
fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label='Training Set', title='Data Train/Test Split')
test.plot(ax=ax, label='Test Set')
ax.axvline('01-01-2015', color='black', ls='--')
ax.legend(['Training Set', 'Test Set'])
plt.show()

# visulize one week
df.loc[(df.index > '01-01-2010') & (df.index < '01-08-2010')
       ].plot(figsize=(15, 5), title='Week Of Data')
plt.show()

#cross validation
#create tss generator.
#n_splits=5 5 fold
#test size is 1 year
#gap 24 hours, ensure we won't cut off a whole day.
tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)
#plot splitted time series
fig, axs = plt.subplots(5, 1, figsize=(15, 15), sharex=True)

fold = 0
for train_idx, val_idx in tss.split(df):
    train = df.iloc[train_idx]
    test = df.iloc[val_idx]
    train['PJME_MW'].plot(ax=axs[fold],
                          label='Training Set',
                          title=f'Data Train/Test Split Fold {fold}')
    test['PJME_MW'].plot(ax=axs[fold],
                         label='Test Set')
    axs[fold].axvline(test.index.min(), color='black', ls='--')
    fold += 1
plt.show()


#create features from timestamp
def create_features(df_in):
    """
    Create time series features based on time series index.
    """
    df = df_in.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

df = create_features(df)

#create lag features
def add_lags(df_in):
    df = df_in.copy()
    #create a dict for later to map
    target_map = df['PJME_MW'].to_dict()
    df['lag1'] = (df.index - pd.Timedelta('364 days')).map(target_map)
    df['lag2'] = (df.index - pd.Timedelta('728 days')).map(target_map)
    df['lag3'] = (df.index - pd.Timedelta('1092 days')).map(target_map)
    return df

df = add_lags(df)

#Train Using Cross Validation
tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)
df = df.sort_index()

fold = 0
preds = []
scores = []
for train_idx, val_idx in tss.split(df):
    train = df.iloc[train_idx]
    test = df.iloc[val_idx]

    train = create_features(train)
    test = create_features(test)

    FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month','year',
                'lag1','lag2','lag3']
    TARGET = 'PJME_MW'

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                           n_estimators=1000,
                           early_stopping_rounds=50,
                           objective='reg:squarederror',
                           max_depth=3,
                           learning_rate=0.01)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100)

    y_pred = reg.predict(X_test)
    preds.append(y_pred)
    score = np.sqrt(mean_squared_error(y_test, y_pred))
    scores.append(score)

print(f'Score across folds {np.mean(scores):0.4f}')
print(f'Fold scores:{scores}')

#optimization

tss = TimeSeriesSplit(n_splits=3, test_size=24*365*1, gap=24)
# ROUND 1
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.01, 0.05],
    'gamma': [0, 0.25, 1.0],
    'reg_lambda': [0, 1.0, 10.0],
}
# Output: max_depth: 4, learning_rate: 0.1, gamma: 0.25, reg_lambda: 10, scale_pos_weight: 3
# Because learning_rate and reg_lambda were at the ends of their range, we will continue to explore those...


# NOTE: To speed up cross validiation, and to further prevent overfitting.
# We are only using a random subset of the data (90%) and are only
# using a random subset of the features (columns) (50%) per tree.

optimal_params = GridSearchCV(
    estimator=xgb.XGBRegressor(objective='reg:squarederror',
                               base_score=0.5,
                               booster='gbtree',
                               n_estimators=500,
                               early_stopping_rounds=10,
                               subsample=0.8,
                               seed = 42),
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error', ## see https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    verbose=0, # NOTE: If you want to see what Grid Search is doing, set verbose=2
    n_jobs = 1,
    cv = tss
)

FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month','year',
            'lag1','lag2','lag3']
TARGET = 'PJME_MW'

X_train = df[FEATURES]
y_train = df[TARGET]
test = df.iloc[df.index >= '01-01-2018']
X_test = test[FEATURES]
y_test = test[TARGET]

optimal_params.fit(X_train,y_train,
                   eval_set=[(X_test, y_test)],
                   verbose=False)


print(optimal_params.best_params_)
print(optimal_params.best_score_)
print(optimal_params.cv_results_['mean_test_score'])

# ROUND 2
tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)
df = df.sort_index()

fold = 0
preds = []
scores = []
for train_idx, val_idx in tss.split(df):
    train = df.iloc[train_idx]
    test = df.iloc[val_idx]

    train = create_features(train)
    test = create_features(test)

    FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month','year',
                'lag1','lag2','lag3']
    TARGET = 'PJME_MW'

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    reg = xgb.XGBRegressor(objective='reg:squarederror',
                               base_score=0.5,
                               booster='gbtree',
                               gamma = 0,
                               learning_rate = 0.05,
                               max_depth = 5,
                               reg_lambda = 1.0,
                               n_estimators=500,
                               early_stopping_rounds=10,
                               seed = 42)

    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100)

    y_pred = reg.predict(X_test)
    preds.append(y_pred)
    score = np.sqrt(mean_squared_error(y_test, y_pred))
    scores.append(score)

print(f'Score across folds {np.mean(scores):0.4f}')
print(f'Fold scores:{scores}')