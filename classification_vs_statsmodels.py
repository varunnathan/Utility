import pandas as pd
import numpy as np

# GLOBALS
local_data_root = '/Users/vnathan/CustomerValueModelling/'
project = local_data_root+'ModelForSettingComp/WithCustomerServicing/'
smf_file = 'segmentation/final_data_segmentation.csv'
class_file = 'experiment_segmented_models/final_prediction_data.csv'
score_dict = {'Score': 'smf', 'predicted_cust_value': 'class'}


def get_cuts(data, bins, target):
    df = data.copy()
    table = df.groupby(bins)[target].min()
    cuts = table.unique().tolist()
    cuts.sort()
    return cuts


def apply_cuts(data, cuts, target):
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


def bin_score(data, score_col):
    df = data.copy()
    dev = df[df['sample'] == 'dev']
    dev.reset_index(drop=True, inplace=True)
    dev[score_col+'_binned'] = pd.qcut(dev[score_col], 10, labels=False)
    cuts = get_cuts(dev, bins=score_col+'_binned', target=score_col)
    df[score_col+'_binned'] = apply_cuts(df, cuts, target=score_col)
    mask = df[score_col+'_binned'] == 0
    df.loc[mask, score_col+'_binned'] = 1
    return df


def perc10(x):
    mask = x.notnull()
    return np.percentile(x[mask], 10)


def perc90(x):
    mask = x.notnull()
    return np.percentile(x[mask], 90)


def rmse(x):
    """
    x: pandas Series with shape (nsamples, )
    """
    return np.sqrt(x.mean())


def group(data, gb, agg_fn):
    df = data.copy()
    cols = agg_fn.keys()
    for col in cols:
        if df[col].isnull().sum() == df.shape[0]:
            _ = agg_fn.pop(col)
    if agg_fn != {}:
        table = df.groupby(gb).agg(agg_fn)
        table.reset_index(inplace=True)
        table.sort(gb, ascending=True, inplace=True)
        return table
    else:
        return None


def display(data, bin_col):
    df = data.copy()
    val = df[df['sample'] == 'val']
    val.reset_index(drop=True, inplace=True)
    DV = 'DV1'
    if bin_col == 'Score_binned':
        pred = 'Score'
        deviance = 'SquaredError_smf'
    else:
        pred = 'predicted_cust_value'
        deviance = 'SquaredError_class'
    agg_pred = {pred: [np.min, np.max, np.nanmean, np.nanmedian]}
    agg_actual = {DV: [np.nanmean, np.nanmedian, perc10, perc90]}
    agg_deviance = {deviance: rmse}
    agg_num_accs = {'ak_concat': len}
    table_pred = group(val, bin_col, agg_pred)
    table_pred.columns = ['Bins', 'Min_Pred', 'Max_Pred', 'Mean_Pred',
                          'Median_Pred']
    table_actual = group(val, bin_col, agg_actual)
    table_actual.columns = ['Bins', 'Mean_Actual', 'Median_Actual',
                            'Perc10_Actual', 'perc90_Actual']
    table_deviance = group(val, bin_col, agg_deviance)
    table_deviance.columns = ['Bins', 'RMSE']
    table_num_accs = group(val, bin_col, agg_num_accs)
    table_num_accs.columns = ['Bins', '#Accounts']
    table_actual.drop('Bins', axis=1, inplace=True)
    table_deviance.drop('Bins', axis=1, inplace=True)
    table_num_accs.drop('Bins', axis=1, inplace=True)
    table = pd.concat([table_pred, table_actual, table_deviance, table_num_accs], axis=1)
    total = ['Total', val[pred].min(), val[pred].max(), val[pred].mean(),
             val[pred].median(), val[DV].mean(),
             val[DV].median(), perc10(val[DV]), perc90(val[DV]),
             rmse(val[deviance]), val.shape[0],
             ]
    table.loc[table.shape[0]] = total
    return table


if __name__ == '__main__':
    print 'comparison between classification model and smf ols reg model...'
    # read data
    smf_pred = pd.read_csv(project+smf_file)
    class_pred = pd.read_csv(project+class_file)
    cols = ['ak_concat', 'sample', 'DV1']
    df = pd.merge(smf_pred[cols+['Score']],
                  class_pred[cols+['predicted_cust_value']], on=cols)
    # bin the score variables
    df1 = bin_score(df, 'Score')
    df2 = bin_score(df1, 'predicted_cust_value')

    for key, value in score_dict.iteritems():
        df2['SquaredError_'+value] = map(lambda x, y: (x-y)**2, df2['DV1'],
                                         df2[key])
    table_smf = display(df2, 'Score_binned')
    table_class = display(df2, 'predicted_cust_value_binned')

    # save
    table_smf.to_csv(project+'smf_results.csv', index=False)
    table_class.to_csv(project+'class_results.csv', index=False)
