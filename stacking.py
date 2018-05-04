import pandas as pd
import numpy as np
import joblib
import json
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf
from dstools.profile import calc_rmse
from sklearn.ensemble import GradientBoostingRegressor  # GBM algorithm
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

# GLOBALS
local_data_root = '/Users/vnathan/CustomerValueModelling/'
project = local_data_root+'ModelForSettingComp/WithCustomerServicing/'
raw_data = project + 'raw_data/'
preprocess_path = project + 'preprocess/'
feat_sel = project + 'feature_selection/'
grid_path = project + 'gridsearch/'
cvs_file = 'cvscores_sc_preprocessed_c60_ols_{}.pkl'
sc_file = 'scores_sc_preprocessed_c60_ols_{}.pkl'


def read_datafile(path):
    fn, ext = os.path.splitext(path)
    read_fn = {'.csv': pd.read_csv,
               '.pkl': joblib.load,
               '.json': json.load}
    return read_fn.get(ext, pd.read_csv)(path)


def add_MKT_predicted(data, ols=True):
    df = data.copy()
    if 'MKT_segment' not in df:
        seg_dict = {'AAA': 'MKT_segment_AAA', 'AAB': 'MKT_segment_AAB',
                    'ABA': 'MKT_segment_ABA', 'ABB': 'MKT_segment_ABB',
                    'BAA': 'MKT_segment_BAA', 'BAB': 'MKT_segment_BAB',
                    'BBA': 'MKT_segment_BBA', 'BBB': 'MKT_segment_BBB'}
        for key, value in seg_dict.iteritems():
            mask = df[value] == 1
            df.loc[mask, 'MKT_segment'] = key
    dev = df[df['sample'] == 'dev']
    dev.reset_index(drop=True, inplace=True)
    if ols:
        table = pd.DataFrame(dev.groupby('MKT_segment')['DV1'].mean())
    else:
        table = pd.DataFrame(dev.groupby('MKT_segment')['DV1'].median())
    table.reset_index(inplace=True)
    val_dict = dict(zip(table['MKT_segment'], table['DV1']))
    df['MKT_predicted'] = df['MKT_segment'].map(val_dict)
    return df


def get_model_params(data, feat_cols):
    """
    fits OLS regression model & returns intercept, model coefficients
    & r-squared
    data: dataset on which model should be fit
    feat_cols: list of features
    """
    df = data.copy()
    # prepare data for modelling
    mask = df['sample'] == 'dev'
    X = df.loc[mask, feat_cols+['DV1']]
    target_col = 'DV1'
    fit_string = '{DV} ~ {IVS}'.format(DV=target_col,
                                       IVS=' + '.join(feat_cols))
    model = smf.ols(fit_string, X)
    res = model.fit()
    pred = res.predict(df)
    params = pd.DataFrame(res.params)
    params.reset_index(inplace=True)
    params.columns = ['feature', 'coef']
    pvalues = pd.DataFrame(res.pvalues)
    pvalues.reset_index(inplace=True)
    pvalues.columns = ['feature', 'pvalue']
    pvalues = pvalues[pvalues['feature'] != 'Intercept']
    pvalues.reset_index(drop=True, inplace=True)
    print res.summary()

    return pred, params, pvalues


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


if __name__ == '__main__':
    print 'stacking is going on...'
    cvscore = {}
    for algo in ['GBR', 'RFR', 'lasso', 'ridge']:
        fn = grid_path + cvs_file.format(algo)
        print 'reading file %s' % (fn)
        cvscore[algo] = read_datafile(fn)
        cvscore[algo].reset_index(inplace=True)
        cvscore[algo].rename(columns={'cvscore': 'cvscore_{}'.format(algo)},
                             inplace=True)

    score = {}
    for algo in ['GBR', 'RFR', 'lasso', 'ridge']:
        fn = grid_path + sc_file.format(algo)
        print 'reading file %s' % (fn)
        score[algo] = read_datafile(fn)
        score[algo].reset_index(inplace=True)
        score[algo].rename(columns={'Score': 'Score_{}'.format(algo)},
                           inplace=True)

    # make a combined dataset
    cols = ['ak_concat', 'sample', 'DV1']
    cvs = pd.merge(pd.merge(cvscore['lasso'], pd.merge(
     cvscore['GBR'], cvscore['RFR'], on=cols), on=cols),
                   cvscore['ridge'], on=cols)
    sc = pd.merge(pd.merge(score['lasso'], pd.merge(
     score['GBR'], score['RFR'], on=cols), on=cols),
                  score['ridge'], on=cols)

    dev = cvs[cvs['sample'] == 'dev']
    dev.reset_index(drop=True, inplace=True)

    val = sc[sc['sample'] == 'val']
    val.reset_index(drop=True, inplace=True)
    cols = list(val.columns)
    cols_new = [x.replace('Score_', 'cvscore_') for x in cols]
    val.rename(columns=dict(zip(cols, cols_new)), inplace=True)

    df = pd.concat([dev[cols_new], val[cols_new]], axis=0)
    df.reset_index(drop=True, inplace=True)

    # add MKT_predicted
    raw = read_datafile(raw_data+'final_modelling_dataset_Oct172017_SC.csv')
    raw = add_MKT_predicted(raw)
    df1 = pd.merge(df, raw[['ak_concat', 'MKT_segment']], on='ak_concat')

    # OLS model using statsmodels
    feat_cols = ['cvscore_lasso', 'cvscore_GBR', 'cvscore_ridge',
                 'cvscore_RFR', 'MKT_segment']
    pred, params, pvalues = get_model_params(data=df1, feat_cols=feat_cols)

    # RMSE on val sample
    mask = df1['sample'] == 'val'
    calc_rmse(df1.loc[mask, 'DV1'], pred[mask])   # 5484.2815561696889

    # GBR model using sklearn
    predictors = ['cvscore_lasso', 'cvscore_GBR', 'cvscore_ridge',
                  'cvscore_RFR', 'MKT_segment_AAB', 'MKT_segment_BBA',
                  'MKT_segment_BBB', 'MKT_segment_BAA', 'MKT_segment_BAB',
                  'MKT_segment_ABA', 'MKT_segment_ABB', 'MKT_segment_AAA']
    df2 = pd.get_dummies(
     df1, prefix='MKT_segment', prefix_sep='_', columns=['MKT_segment'])
    gbm0 = GradientBoostingRegressor(random_state=4, n_estimators=250,
                                     learning_rate=0.03, max_depth=3,
                                     max_features='sqrt')
    train = df2[df2['sample'] == 'dev']
    train.reset_index(drop=True, inplace=True)
    test = df2[df2['sample'] == 'val']
    test.reset_index(drop=True, inplace=True)
    modelfit(gbm0, train, test, predictors, target='DV1')  # 5485.719346

    # parameter tuning
    param_test1 = {'learning_rate': [0.001, 0.005, 0.009, 0.01, 0.02]}
    gsearch1 = GridSearchCV(estimator=GradientBoostingRegressor(
     min_samples_leaf=65, min_samples_split=30, learning_rate=0.01,
     max_depth=2, max_features='sqrt', subsample=0.8, n_estimators=280,
     random_state=4), param_grid=param_test1, scoring='r2', iid=False, cv=4)
    gsearch1.fit(train[predictors], train['DV1'])

    gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

    modelfit(gsearch1.best_estimator_, train, test, predictors, target='DV1')
    # 5478.909473

    gbm_tuned = GradientBoostingRegressor(
     learning_rate=0.001, n_estimators=5000, max_depth=1,
     min_samples_split=2, min_samples_leaf=65, subsample=0.5,
     random_state=4, max_features='sqrt')
    modelfit(gbm_tuned, train, test, predictors, target='DV1')
    # 5478.366498
