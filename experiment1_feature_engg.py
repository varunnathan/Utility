import pandas as pd
import joblib
import numpy as np
import json
import os
import pickle
from dstools.profile import calc_rmse
from sklearn.ensemble import GradientBoostingRegressor  # GBM algorithm
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
import statsmodels.api as sm
import statsmodels.formula.api as smf

# GLOBALS
local_data_root = '/Users/vnathan/CustomerValueModelling/'
project = local_data_root+'ModelForSettingComp/WithCustomerServicing/'
raw_data = project + 'raw_data/'
preprocess_path = project + 'preprocess/'
feat_sel = project + 'feature_selection/'
grid_path = project + 'gridsearch/'
inp_file = 'final_modelling_dataset_Oct172017_SC_preprocessed_c60.pkl'


def read_datafile(path):
    fn, ext = os.path.splitext(path)
    read_fn = {'.csv': pd.read_csv,
               '.pkl': joblib.load,
               '.json': json.load}
    return read_fn.get(ext, pd.read_csv)(path)


def modelfit(alg, dtrain, dtest, predictors, target, performCV=True,
             printFeatureImportance=True, cv_folds=4):
    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    # Predict on testing set:
    dtest_predictions = alg.predict(dtest[predictors])

    # Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(
         alg, dtrain[predictors], dtrain[target], cv=cv_folds, scoring='r2')

    # Print model report:
    print "\n Model Report"
    print "Dev R2 : %.4g" % metrics.r2_score(
     dtrain[target].values, dtrain_predictions)
    print "Val R2 : %.4g" % metrics.r2_score(
     dtest[target].values, dtest_predictions)
    print "Dev RMSE : %f" % calc_rmse(dtrain[target], dtrain_predictions)
    print "Val RMSE : %f" % calc_rmse(dtest[target], dtest_predictions)

    if performCV:
        print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score))


def get_sample_data_for_model(data):
    df = data.copy()
    mask = df['sample'] == 'val'
    val = df.loc[mask, :]
    val.reset_index(drop=True, inplace=True)
    dev = df.loc[~mask, :]
    dev.reset_index(drop=True, inplace=True)

    return dev, val


def get_best_feats_for_experiment(data, model):
    test, train = get_sample_data_for_model(data)
    cols = list(train.columns)
    for col in ['sample', 'DV1', 'ak_concat']:
        cols.remove(col)
    gbm0 = GradientBoostingRegressor(random_state=4, n_estimators=250,
                                     learning_rate=0.03, max_depth=3,
                                     max_features='sqrt')
    print modelfit(gbm0, train, test, predictors=cols, target='DV1')
    feat_imp_val = pd.DataFrame(
     {'feature': cols, 'importance': gbm0.feature_importances_})
    cols.sort()
    feat_imp_dev = pd.DataFrame(
     {'feature': cols, 'importance': model.feature_importances_})
    feat_imp_val.sort('importance', ascending=False, inplace=True)
    feat_imp_dev.sort('importance', ascending=False, inplace=True)
    feat_imp_val.reset_index(drop=True, inplace=True)
    feat_imp_dev.reset_index(drop=True, inplace=True)
    best_50_feats_dev = feat_imp_dev.loc[:49, 'feature'].unique().tolist()
    best_50_feats_val = feat_imp_val.loc[:49, 'feature'].unique().tolist()
    best_feats = best_50_feats_dev + list(
     set(best_50_feats_val) - set(best_50_feats_dev))
    return best_feats


def get_model_params(data, feat_cols, new_data):
    """
    fits OLS regression model & returns intercept, model coefficients
    & r-squared
    data: dataset on which model should be fit
    feat_cols: list of features
    """
    df = data.copy()
    df1 = new_data.copy()
    # prepare data for modelling
    mask = df['sample'] == 'dev'
    X = df.loc[mask, feat_cols+['DV1']]
    target_col = 'DV1'
    fit_string = '{DV} ~ {IVS}'.format(DV=target_col,
                                       IVS=' + '.join(feat_cols))
    model = smf.ols(fit_string, X)
    res = model.fit()
    pred = res.predict(df1)
    params = pd.DataFrame(res.params)
    params.reset_index(inplace=True)
    params.columns = ['feature', 'coef']
    pvalues = pd.DataFrame(res.pvalues)
    pvalues.reset_index(inplace=True)
    pvalues.columns = ['feature', 'pvalue']
    pvalues = pvalues[pvalues['feature'] != 'Intercept']
    pvalues.reset_index(drop=True, inplace=True)
    print res.summary()

    return pred, params, pvalues, res


if __name__ == '__main__':
    print 'experiment 1:'
    # read data
    df = read_datafile(feat_sel+inp_file)
    df1 = df.copy()
    model = read_datafile(grid_path+'model_sc_preprocessed_c60_ols_GBR.pkl')

    best_feats = get_best_feats_for_experiment(data=df, model=model)

    gbm0 = GradientBoostingRegressor(random_state=4, n_estimators=250,
                                     learning_rate=0.03, max_depth=3,
                                     max_features='sqrt')
    train, test = get_sample_data_for_model(df)
    print modelfit(gbm0, train, test, predictors=best_feats, target='DV1')
    feat_imp = pd.DataFrame(
     {'feature': best_feats, 'importance': gbm0.feature_importances_})
    feat_imp.sort('importance', ascending=False, inplace=True)

    # look at each feature individually
    # remove features with different signs of correlation coefficient
    corr_dev = []
    corr_val = []
    for col in best_feats:
        corr_dev.append(np.corrcoef(train[col], train['DV1'])[0, 1])
        corr_val.append(np.corrcoef(test[col], test['DV1'])[0, 1])
    corr = pd.DataFrame(
     {'feature': best_feats, 'dev': corr_dev, 'val': corr_val})
    corr['remove'] = map(lambda x, y: 1 if np.sign(x) != np.sign(y) else 0,
                         corr['dev'], corr['val'])
    best_feats1 = corr['feature'][corr['remove'] == 0].unique().tolist()
    # 47 features
    # build SMF OLS model
    pred, params, pvalues = get_model_params(df, feat_cols=best_feats1)
    mask = df['sample'] == 'dev'
    print calc_rmse(df.loc[mask, 'DV1'], pred[mask])  # 5087.61725855
    print calc_rmse(df.loc[~mask, 'DV1'], pred[~mask])  # 5475.00055475

    feats_significant = pvalues['feature'][
     pvalues['pvalue'] <= 0.05].unique().tolist()
    pred, params, pvalues = get_model_params(df, feat_cols=feats_significant)
    mask = df['sample'] == 'dev'
    print calc_rmse(df.loc[mask, 'DV1'], pred[mask])  # 5091.11691215
    print calc_rmse(df.loc[~mask, 'DV1'], pred[~mask])  # 5483.08508696

    # build a model without certain observation points from dev sample
    # drop dev vintages - 2015-02-28, 2015-03-31, 2015-05-31, 2015-06-30, 2016-05-31
    df1['date'] = df1['ak_concat'].apply(lambda x: x.split('_')[1])
    mask = (df1['date'].isin(['2015-02-28', '2015-03-31', '2015-05-31',
                              '2015-06-30', '2016-05-31', '2015-07-31']) &
            (df1['sample'] == 'dev'))
    df2 = df1.loc[~mask, :]
    df2.reset_index(drop=True, inplace=True)

    pred, params, pvalues, model = get_model_params(df2, feat_cols=best_feats1, new_data=df1)
    mask = df1['sample'] == 'dev'
    print calc_rmse(df1.loc[mask, 'DV1'], pred[mask])  # 5095.42509531
    print calc_rmse(df1.loc[~mask, 'DV1'], pred[~mask])  # 5469.70733356
    # best model
    # save
    pickle.dump(model, open(grid_path+'best_smf_ols_model.pkl', 'w'))
