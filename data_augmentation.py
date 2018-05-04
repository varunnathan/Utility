import pandas as pd
import numpy as np
import joblib
import json
import os

# GLOBALS
local_data_root = '/Users/vnathan/CustomerValueModelling/'
project = local_data_root+'ModelForSettingComp/WithCustomerServicing/'
raw_data = project + 'raw_data/'
preprocess_path = project + 'preprocess/'
feat_sel = project + 'feature_selection/'
grid_path = project + 'gridsearch/'
inp_file = 'final_modelling_dataset_Oct172017_SC_preprocessed_c60.pkl'
score_file = 'scores_sc_preprocessed_c60_ols_GBR.pkl'
out_file = 'final_modelling_dataset_Oct172017_SC_preprocessed_c60_augmented_r{}.pkl'


def read_datafile(path):
    fn, ext = os.path.splitext(path)
    read_fn = {'.csv': pd.read_csv,
               '.pkl': joblib.load,
               '.json': json.load}
    return read_fn.get(ext, pd.read_csv)(path)


def augment_dev_data(data, scores, sample_rate=0.3, out=None):
    df = data.copy()
    score = scores.copy()
    score.reset_index(inplace=True)
    dev = df[df['sample'] == 'dev']
    val = df[df['sample'] == 'val']
    dev.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    num_samples = int(round(sample_rate*len(val)))
    val1 = val.sample(n=num_samples)
    psuedo_val_labels = score[score['sample'] == 'val'][['ak_concat', 'Score']]
    psuedo_val_labels.reset_index(drop=True, inplace=True)
    val1.reset_index(drop=True, inplace=True)
    val1['DV1'] = pd.merge(val1, psuedo_val_labels, on='ak_concat')['Score']
    val1['sample'] = 'dev'
    cols = list(dev.columns)
    final = pd.concat([dev[cols], val1[cols], val[cols]], axis=0)
    final.reset_index(drop=True, inplace=True)
    final['ak_concat1'] = range(len(final))

    if out:
        final.to_pickle(out)
    else:
        return final


def grid(inp, jobname):
    cmd = 'kbglearn gridsearch -a gbr --python 2 --cv-folds 4 --random-state 54545454 --index-col ak_concat1 --feat-groups CR,EA,FICO,KH,LI,MI,PAH,PP,RV,SC,YAH,YL,MKT --sample-col sample --target-col DV1 --datafile {} {} --queue'.format(inp, jobname)
    os.system(cmd)


if __name__ == '__main__':
    print 'psuedo labelling is running...'
    # read data
    df = read_datafile(feat_sel+inp_file)
    scores = read_datafile(grid_path+score_file)
    # create augmented datasets
    for prop in [0.2, 0.3, 0.4, 0.5]:
        print prop
        name = int(prop*100)
        augment_dev_data(data=df, scores=scores, sample_rate=prop,
                         out=feat_sel+out_file.format(name))

    # gridsearch
    for prop, jn in [(20, 'augmented_r20'), (30, 'augmented_r30'),
                     (40, 'augmented_r40'), (50, 'augmented_r50')]:
        inp = feat_sel+out_file.format(prop)
        jobname = 'sc_preprocessed_c60_ols_'+jn
        grid(inp=inp, jobname=jobname)
