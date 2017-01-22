import pandas
from sklearn.ensemble import GradientBoostingClassifier
import evaluation
import os
import xgboost


train = pandas.read_csv('training.csv', index_col='id')


variables = list(train.columns[1:-5])

params = {"objective": "binary:logistic",
          "eta": 0.3,
          "max_depth": 5,
          "min_child_weight": 3,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 1}

plst = list(params.items())


Xdatatrain = xgboost.DMatrix(data = train[variables], label = train['signal'])
bst = xgboost.train(plst, Xdatatrain)

check_agreement = pandas.read_csv('check_agreement.csv', index_col='id')
agreement_probs = bst.predict(xgboost.DMatrix(check_agreement[variables]))


ks = evaluation.compute_ks(
    agreement_probs[check_agreement['signal'].values == 0],
    agreement_probs[check_agreement['signal'].values == 1],
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)
print 'KS me tric', ks, ks < 0.09

check_correlation = pandas.read_csv('check_correlation.csv', index_col='id')
correlation_probs = bst.predict(xgboost.DMatrix(check_correlation[variables]))

cvm = evaluation.compute_cvm(correlation_probs, check_correlation['mass'])
print 'CvM metric', cvm, cvm < 0.002

train_eval = train[train['min_ANNmuon'] > 0.4]
train_probs = bst.predict(xgboost.DMatrix(train_eval[variables]))
AUC = evaluation.roc_auc_truncated(train_eval['signal'], train_probs)
print 'AUC', AUC