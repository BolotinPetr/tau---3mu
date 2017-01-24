import pandas as pd
import evaluation
import xgboost
import copy
from sklearn.ensemble import RandomForestClassifier


def add_features(df):
    df['flight_dist_sig'] = df['FlightDistance']/df['FlightDistanceError']
    df['NEW_FD_SUMP']=df['FlightDistance']/(df['p0_p']+df['p1_p']+df['p2_p'])
    df['NEW5_lt']=df['LifeTime']*(df['p0_IP']+df['p1_IP']+df['p2_IP'])/3
    df['p_track_Chi2Dof_MAX'] = df.loc[:, ['p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof']].max(axis=1)
    df['p0p2_ip_ratio']=df['IP']/df['IP_p0p2']
    df['p1p2_ip_ratio']=df['IP']/df['IP_p1p2']
    df['DCA_MAX'] = df.loc[:, ['DOCAone', 'DOCAtwo', 'DOCAthree']].max(axis=1)
    df['iso_bdt_min'] = df.loc[:, ['p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT']].min(axis=1)
    df['iso_min'] = df.loc[:, ['isolationa', 'isolationb', 'isolationc','isolationd', 'isolatione', 'isolationf']].min(axis=1)
    return df

train = pd.read_csv('training.csv', index_col='id')
test = pd.read_csv('test.csv')
check_agreement = pd.read_csv('check_agreement.csv', index_col='id')
check_correlation = pd.read_csv('check_correlation.csv', index_col='id')

bad_features = list(train.columns[-5:])
train = add_features(train)
test = add_features(test)
check_correlation = add_features(check_correlation)
check_agreement = add_features(check_agreement)

variables = list(train.columns[:])
variables_copy = copy.copy(variables)
for c_name in variables_copy:
    for bf_name in bad_features:
        if c_name == bf_name:
            c_copy = copy.copy(c_name)
            variables.remove(c_name)
print variables

params = {"objective": "binary:logistic",
          "eta": 0.3,
          "max_depth": 5,
          "min_child_weight": 3,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 1}

plst = list(params.items())
num_trees = 250

Xdatatrain = xgboost.DMatrix(data = train[variables], label = train['signal'])
bst = xgboost.train(plst, Xdatatrain, num_trees)

agreement_probs = bst.predict(xgboost.DMatrix(check_agreement[variables]))


ks = evaluation.compute_ks(
    agreement_probs[check_agreement['signal'].values == 0],
    agreement_probs[check_agreement['signal'].values == 1],
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)
print 'KS me trick', ks, ks < 0.09

correlation_probs = bst.predict(xgboost.DMatrix(check_correlation[variables]))

cvm = evaluation.compute_cvm(correlation_probs, check_correlation['mass'])
print 'CvM metric', cvm, cvm < 0.002

train_eval = train[train['min_ANNmuon'] > 0.4]
train_probs = bst.predict(xgboost.DMatrix(train_eval[variables]))
AUC = evaluation.roc_auc_truncated(train_eval['signal'], train_probs)
print 'AUC', AUC


rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(train[variables], train["signal"])
test_probs = (rf.predict_proba(test[variables])[:,1] +
              bst.predict(xgboost.DMatrix(test[variables])))/2
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("xgboost_submission.csv", index=False)
