import pandas as pd
import numpy as np
import joblib
import json
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf
from dstools.profile import calc_rmse

# GLOBALS
local_data_root = '/Users/vnathan/CustomerValueModelling/'
project = local_data_root+'ModelForSettingComp/WithCustomerServicing/'
raw_data = project + 'raw_data/'
preprocess_path = project + 'preprocess/'
feat_sel = project + 'feature_selection/'
grid_path = project + 'gridsearch/'
inp_file = 'final_modelling_dataset_Oct172017_SC.csv'


def read_datafile(path):
    fn, ext = os.path.splitext(path)
    read_fn = {'.csv': pd.read_csv,
               '.pkl': joblib.load,
               '.json': json.load}
    return read_fn.get(ext, pd.read_csv)(path)


def create_validation_set_for_stacking(
 data, random_state=12121212, split=0.75):
    df = data.copy()
    np.random.seed(random_state)
    accs = df[df['sample'] == 'dev']['accountkey'].unique().tolist()
    accs1 = np.random.permutation(accs)
    num_accs_dev = int(round(split*len(accs1), 0))
    accs_oot = accs1[num_accs_dev:]
    mask = df['accountkey'].isin(accs_oot)
    df.loc[mask, 'sample'] = 'oot'
    return df


def preprocess(inp, out, pipe):
    cmd = 'kbglearn preprocess inf2nan exrsquaredimputer exoutscaler stdscaler --python 2 --target-col DV1 --feat-groups CR,EA,FICO,KH,LI,MI,PAH,PP,RV,SC,YAH,YL,MKT  --sample-col sample --index-col ak_concat --datafile {} --output-fname {}; mv pipeline.pkl {}'.format(inp, out, pipe)
    os.system(cmd)


def feat_selection(inp, out, corr_coef=0.5):
    cmd = 'kbglearn featsel-cluster-contDV -c {} --python 2 --index-col ak_concat --feat-groups CR,EA,FICO,KH,LI,MI,PAH,PP,RV,SC,YAH,YL,MKT --target-col DV1 --sample-col sample --datafile {}; mv data.pkl {}'.format(corr_coef, inp, out)
    os.system(cmd)


def grid(inp, jobname):
    cmd = 'kbglearn gridsearch -a ridge,lasso,gbr,rfr --python 2 --cv-folds 4 --random-state 54545454 --index-col ak_concat --feat-groups CR,EA,FICO,KH,LI,MI,PAH,PP,RV,SC,YAH,YL,MKT --sample-col sample --target-col DV1 --datafile {} {} --queue'.format(inp, jobname)
    os.system(cmd)


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


def get_model_params(q, data, feat_cols):
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


if __name__ == '__main__':
    print 'preparing dataset for stacking experiment...'
    # read the dataset
    df = read_datafile(raw_data + inp_file)
    # add MKT_segment to df
    tmp = read_datafile(local_data_root+'ModelForSettingComp/raw_data/final_modelling_dataset_Oct62017_newDV.csv')
    df = pd.merge(df, tmp[['ak_concat', 'MKT_segment']], on='ak_concat')
    # One-hot encode MKT_segment
    df = pd.get_dummies(df, prefix='MKT_segment', prefix_sep='_',
                        columns=['MKT_segment'])
    # add oot sample
    df1 = create_validation_set_for_stacking(df)

    # save df and df1
    df.to_csv(raw_data+inp_file, index=False)
    df1.to_csv(raw_data+'final_modelling_dataset_Oct172017_SC_with_oot.csv',
               index=False)

    # preprocess
    # without oot
    inp = raw_data + inp_file
    out = preprocess_path + 'final_modelling_dataset_Oct172017_SC_preprocessed.pkl'
    pipe = preprocess_path + 'pipeline_modelling_dataset_Oct172017_SC.pkl'
    preprocess(inp, out, pipe)

    # with oot
    inp = raw_data + 'final_modelling_dataset_Oct172017_SC_with_oot.csv'
    out = preprocess_path + 'final_modelling_dataset_Oct172017_SC_with_oot_preprocessed.pkl'
    pipe = preprocess_path + 'pipeline_modelling_dataset_Oct172017_SC_with_oot.pkl'
    preprocess(inp, out, pipe)

    # feature selection
    # without oot
    inp = preprocess_path + 'final_modelling_dataset_Oct172017_SC_preprocessed.pkl'
    out1 = feat_sel + 'final_modelling_dataset_Oct172017_SC_preprocessed_c50.pkl'
    out2 = feat_sel + 'final_modelling_dataset_Oct172017_SC_preprocessed_c60.pkl'
    feat_selection(inp, out1)
    feat_selection(inp, out2, corr_coef=0.6)

    # with oot
    inp = preprocess_path + 'final_modelling_dataset_Oct172017_SC_with_oot_preprocessed.pkl'
    out1 = feat_sel + 'final_modelling_dataset_Oct172017_SC_with_oot_preprocessed_c50.pkl'
    out2 = feat_sel + 'final_modelling_dataset_Oct172017_SC_with_oot_preprocessed_c60.pkl'
    feat_selection(inp, out1)
    feat_selection(inp, out2, corr_coef=0.6)

    # gridsearch
    for x, y, z in [
     ('preprocessed', '50', 'sc_preprocessed_c50_ols'),
     ('preprocessed', '60', 'sc_preprocessed_c60_ols'),
     ('with_oot_preprocessed', '50', 'sc_with_oot_preprocessed_c50_ols'),
     ('with_oot_preprocessed', '60', 'sc_with_oot_preprocessed_c60_ols')]:
        inp = feat_sel + 'final_modelling_dataset_Oct172017_SC_{}_c{}.pkl'.format(x, y)
        jobname = z
        print 'jobname: %s' % (jobname)
        grid(inp=inp, jobname=jobname)

    # topmodels
    # MKT_segment
    # val = 5498.2871290034291 and dev = 5115.2104709054229
