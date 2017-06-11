# Linear Regression
import pandas
import numpy as np
import gc
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df_train = pandas.read_csv('train_2016_temp_full_number_ex_propertyzoningdesc_propertycountylandusecode.csv', sep='|')
df_train = df_train.drop(['propertyzoningdesc', 'propertycountylandusecode'], axis=1).fillna(0)

X = df_train.drop(['logerror'], axis=1).values
X = X.astype('float32')
Y = df_train['logerror'].values
Y = Y.astype('float32')

del df_train; gc.collect()

'''
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = LinearRegression()
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())
'''
'''
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = DecisionTreeRegressor()
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())
'''
'''
seed = 7
# Spot Check Algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('Rid', Ridge()))
models.append(('Lasso', Lasso()))
models.append(('ENet', ElasticNet()))
models.append(('KNR', KNeighborsRegressor()))
models.append(('DeTree', DecisionTreeRegressor()))
models.append(('SVR', SVR()))
# evaluate each model in turn
results = []
names = []
scoring = 'neg_mean_squared_error'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
#LR: -0.025940 (0.003163)
#Rid: -0.025844 (0.003160)
#ENet: -0.025850 (0.003148)
#KNR: -0.030024 (0.002767)
#DeTree: -0.057362 (0.004170)
#SVR: lau
'''
'''
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# fit model no training data
model = LinearRegression()
model.fit(X_train, y_train)
print(model)

# make predictions for test data
y_pred = model.predict(X_test)
print('The mse of prediction is:', mean_squared_error(y_test, y_pred))
 #0.0246500929513
 '''
import lightgbm as lgb
from lightgbm import plot_importance
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

'''
print('Start training...')
# train
gbm = lgb.LGBMRegressor(objective='regression', num_leaves=31,learning_rate=0.05,n_estimators=20)
gbm.fit(X_train, y_train,eval_set=[(X_test, y_test)], eval_metric='l1',early_stopping_rounds=5)

print('Start predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print('The mse of prediction is:', mean_squared_error(y_test, y_pred))

print('Calculate feature importances...')
# feature importances
print('Feature importances:', list(gbm.feature_importances_))
plot_importance(gbm)
pyplot.show()
'''


# other scikit-learn modules
estimator = lgb.LGBMRegressor()

param_grid = {	
    'learning_rate': [0.01, 0.1, 1,0.0021],
    'n_estimators': [20, 40],
	'num_leaves' : [31,200,500,512]
}

gbm = GridSearchCV(estimator, param_grid)
gbm.fit(X_train, y_train)
print(gbm)
print('Best parameters found by grid search are:', gbm.best_params_)
#0.1,20,31