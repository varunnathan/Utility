import pandas as pd
import joblib
import numpy as np
import json
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples
import statsmodels.api as sm
import statsmodels.formula.api as smf
from collections import defaultdict
from dstools.profile import calc_rmse
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor  # GBM algorithm
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

# GLOBALS
local_data_root = '/Users/vnathan/CustomerValueModelling/'
project = local_data_root+'ModelForSettingComp/WithCustomerServicing/'
raw_data = project + 'raw_data/'
experiment = project + 'experiment_segmented_models/regression/'
inp_file = 'final_modelling_dataset_Oct172017_SC.csv'
out_file = 'final_modelling_dataset_Oct172017_SC_segment{}.csv'
preprocess_out_file = 'final_modelling_dataset_Oct172017_SC_segment{}_preprocessed.pkl'
featsel_out_file = 'final_modelling_dataset_Oct172017_SC_segment{}_preprocessed_c60.pkl'
pipe_out_file = 'pipeline_modelling_dataset_Oct172017_SC_segment{}.pkl'
job = 'sc_preprocessed_c60_ols_segment{}'
segment_cols = 'DV1'
feat_groups = ('CR', 'EA', 'FICO', 'KH', 'LI', 'MI', 'PAH', 'PP', 'RV', 'SC',
               'YAH', 'YL', 'MKT')
models = {1: '35f6e71aR', 2: 'bda63526R', 3: '74c01da3R', 4: '255454beR',
          5: '6fd5bda0R'}
model_inp_file = [experiment+featsel_out_file.format(x) for x in range(1, 6)]
model_file = '/Users/vnathan/.qapla/models/{}/model.pkl'
model_pipe_file = [experiment+pipe_out_file.format(x) for x in range(1, 6)]
model_dict = defaultdict(list)
for item in range(1, 6):
    model_dict['segment'+str(item)].append(
     experiment+featsel_out_file.format(item))
    model_dict['segment'+str(item)].append(
     experiment+pipe_out_file.format(item))
    model_dict['segment'+str(item)].append(model_file.format(models[item]))


def read_datafile(path):
    fn, ext = os.path.splitext(path)
    read_fn = {'.csv': pd.read_csv,
               '.pkl': joblib.load,
               '.json': json.load}
    return read_fn.get(ext, pd.read_csv)(path)


def get_cuts(data, bins='DV1_qcut', target='DV1'):
    df = data.copy()
    table = df.groupby(bins)[target].min()
    cuts = table.unique().tolist()
    cuts.sort()
    return cuts


def apply_cuts(data, cuts, target='DV1'):
    df = data.copy()
    temp = df[target]
    binned = pd.Series([-2] * len(temp), index=temp.index)
    binned[temp.isnull()] = -1
    binned[temp < np.min(cuts)] = 0

    for ibin, (low, high) in enumerate(zip(cuts[:-1], cuts[1:])):
        mask = (temp >= low) & (temp < high)
        binned[mask] = ibin + 1
    binned[temp >= np.max(cuts)] = len(cuts)
    return binned


def preprocess(inp, out, pipe):
    cmd = 'kbglearn preprocess inf2nan exrsquaredimputer exoutscaler stdscaler --python 2 --target-col DV1 --feat-groups CR,EA,FICO,KH,LI,MI,PAH,PP,RV,SC,YAH,YL,MKT  --sample-col sample --index-col ak_concat --datafile {} --output-fname {}; mv pipeline.pkl {}'.format(inp, out, pipe)
    os.system(cmd)


def feat_selection(inp, out, corr_coef=0.6):
    cmd = 'kbglearn featsel-cluster-contDV -c {} --python 2 --index-col ak_concat --feat-groups CR,EA,FICO,KH,LI,MI,PAH,PP,RV,SC,YAH,YL,MKT --target-col DV1 --sample-col sample --datafile {}; mv data.pkl {}'.format(corr_coef, inp, out)
    os.system(cmd)


def grid(inp, jobname):
    cmd = 'kbglearn gridsearch -a ridge,lasso,gbr,rfr --python 2 --cv-folds 4 --random-state 54545454 --index-col ak_concat --feat-groups CR,EA,FICO,KH,LI,MI,PAH,PP,RV,SC,YAH,YL,MKT --sample-col sample --target-col DV1 --datafile {} {} --queue'.format(inp, jobname)
    os.system(cmd)


def score(data, model_dict):
    """
    data: dataset to be scored
    """
    df = data.copy()
    cols_needed = ['ak_concat', 'sample', 'DV1']
    for key, values in model_dict.iteritems():
        print key
        feats = read_datafile(values[0])
        pipe = read_datafile(values[1])
        model = read_datafile(values[2])
        feat_cols = [x for x in list(feats.columns) if x.startswith(
         feat_groups)]
        print 'generate feature importance file:'
        feat_cols.sort()
        feat_imp = pd.DataFrame(
         {'feature': feat_cols, 'importance': model.feature_importances_})
        feat_imp = feat_imp[feat_imp['importance'] != 0]
        feat_imp.reset_index(drop=True, inplace=True)
        feat_imp.sort('importance', ascending=False, inplace=True)
        feat_imp.to_csv(
         experiment+'feat_importance_{}.csv'.format(key), index=False)
        print 'generate score file:'
        X = df[pipe.feature_names].values
        X_pre = pipe.transform(X)
        df_pre = pd.DataFrame(X_pre, columns=pipe.feature_names)
        score = model.predict(df_pre[feat_cols])
        df['score_{}'.format(key)] = score
        df[cols_needed+['score_{}'.format(key)]].to_csv(
         experiment+'scores_{}.csv'.format(key), index=False)


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


def choose_model_with_best_perf(data, feat_cols, target='DV1'):
    df = data.copy()
    for col in feat_cols:
        df[col+'_squared_error'] = map(lambda x, y: (x-y)**2, df[target],
                                       df[col])
    # choose score that outputs the least squared error
    error_cols = [
     'score_segment{}_squared_error'.format(i) for i in range(1, 6)]
    df['least_squared_error'] = df[error_cols].min(axis=1)
    for i in range(1, 6):
        mask = df['least_squared_error'] == df[
         'score_segment{}_squared_error'.format(i)]
        df.loc[mask, 'final_score'] = df.loc[mask, 'score_segment{}'.format(i)]

    return df


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
    print 'prepare segments for modelling...'
    # read data
    df = read_datafile(raw_data+inp_file)
    mask = df['sample'] == 'dev'
    # X = df.loc[mask, segment_cols].values
    # X = X.reshape((X.shape[0], 1))
    '''
    # elbow method to find optimal n_clusters
    distortions = []
    for i in range(2, 11):
        print i
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=0)
        km.fit(X)
        distortions.append(km.inertia_)

    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.show()

    # optimum # clusters = 6
    km = KMeans(n_clusters=6, init='k-means++', n_init=10, random_state=0)
    km.fit(X)
    X_overall = df[segment_cols].values
    X_overall = X_overall.reshape((X_overall.shape[0], 1))
    y_km = km.predict(X_overall)
    df['cluster_label'] = y_km

    # silhouette coefficient
    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(X_overall, y_km, metric='euclidean')
    '''
    dev = df.loc[mask, :]
    dev.reset_index(drop=True, inplace=True)
    dev['DV1_qcut'] = pd.qcut(dev['DV1'], q=5, labels=False)
    cuts = get_cuts(dev)
    df['DV1_cut'] = apply_cuts(df, cuts)
    val = df.loc[~mask, :]
    val.reset_index(drop=True, inplace=True)
    cols = list(df.columns)

    # save
    for item in df['DV1_cut'].unique().tolist():
        print item
        mask = (df['DV1_cut'] == item)
        df1 = df[mask]
        df1.reset_index(drop=True, inplace=True)
        df1.to_csv(experiment+out_file.format(item), index=False)

    # preprocess
    for item in range(1, 6):
        print item
        inp = experiment+out_file.format(item)
        out = experiment+preprocess_out_file.format(item)
        pipe = experiment+pipe_out_file.format(item)
        preprocess(inp, out, pipe)

    # feature selection
    for item in range(1, 6):
        inp = experiment+preprocess_out_file.format(item)
        out = experiment+featsel_out_file.format(item)
        feat_selection(inp, out)

    # gridsearch
    for item in range(1, 6):
        inp = experiment+featsel_out_file.format(item)
        jobname = job.format(item)
        print 'jobname: %s' % (jobname)
        grid(inp=inp, jobname=jobname)

    # get final results
    score(df, model_dict)

    # build final model
    # combine data
    score = read_datafile(experiment+'scores_segment{}.csv'.format(1))
    for i in range(2, 6):
        tmp = read_datafile(experiment+'scores_segment{}.csv'.format(i))
        score = pd.merge(score, tmp[['ak_concat', 'score_segment{}'.format(
         i)]], on='ak_concat')

    score.to_csv(experiment+'final_scores_from_all_segments.csv', index=False)

    feat_cols = ['score_segment{}'.format(x) for x in range(1, 6)]
    # least squares model
    least_squares_model = choose_model_with_best_perf(score, feat_cols=feat_cols)

    # kbglearn models
    inp = experiment+'final_scores_from_all_segments.csv'
    jobname = 'combined_score_ols'
    cmd = 'kbglearn gridsearch -a ridge,lasso,gbr,rfr --python 2 --cv-folds 4 --random-state 54545454 --index-col ak_concat --feat-groups score --sample-col sample --target-col DV1 --datafile {} {} --queue'.format(inp, jobname)
    os.system(cmd)

    # assess performance
    mask = final_score['sample'] == 'dev'
    print calc_rmse(final_score.loc[mask, 'DV1'],
                    final_score.loc[mask, 'final_score'])
    print calc_rmse(final_score.loc[~mask, 'DV1'],
                    final_score.loc[~mask, 'final_score'])
    print r2_score(final_score.loc[mask, 'DV1'],
                    final_score.loc[mask, 'final_score'])
    print r2_score(final_score.loc[~mask, 'DV1'],
                    final_score.loc[~mask, 'final_score'])
