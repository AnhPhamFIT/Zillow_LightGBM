import pandas
import numpy as np
import gc
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

df_train = pandas.read_csv('train_2016_temp_full_number_ex_propertyzoningdesc_propertycountylandusecode.csv', sep='|')
df_train = df_train.drop(['propertyzoningdesc', 'propertycountylandusecode'], axis=1).fillna(0)

X = df_train.drop(['logerror'], axis=1).values
X = X.astype('float32')
Y = df_train['logerror'].values
Y = Y.astype('float32')

del df_train; gc.collect()

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)



d_train = lgb.Dataset(X_train, label=y_train)
d_valid = lgb.Dataset(X_test, label=y_test)

'''
params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.0021 # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l2'          # or 'mse':12 ,'mae':11
params['sub_feature'] = 0.5      # feature_fraction 
params['bagging_fraction'] = 0.85 # sub_row
params['bagging_freq'] = 40
params['num_leaves'] = 512        # num_leaf
params['min_data'] = 500         # min_data_in_leaf
params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf

num_round = 400
clf = lgb.train(params, d_train, num_round, valid_sets=[d_valid])
'''

params = {}
params['max_bin'] = 20
params['learning_rate'] = 0.003 # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l1'          # or 'mse':12 ,'mae':11
#params['sub_feature'] = 0.5      # feature_fraction 
#params['bagging_fraction'] = 0.85 # sub_row
#params['bagging_freq'] = 40
params['num_leaves'] = 512        # num_leaf
params['min_data'] = 500         # min_data_in_leaf
#params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
#params['num_trees'] = 20         #n_estimators

clf = lgb.train(params, d_train, early_stopping_rounds=10, valid_sets=[d_valid])
print("Start prediction ...")
# num_threads > 1 will predict very slow in kernal
clf.reset_parameter({"num_threads":1})
#y_pred = clf.predict(X_test) - old
y_pred = clf.predict(X_test,num_iteration=clf.best_iteration)
#mse 
#mse = mean_squared_error(y_test, y_pred)
#print("MSE: %f" % (mse))
mae = mean_absolute_error(y_test, y_pred)
print("MAE: %f" % (mae))