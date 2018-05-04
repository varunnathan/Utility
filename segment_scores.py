import pandas as pd
import numpy as np
import joblib
import json
import os
from segmentation_one_variable_continuousDV_RMSE import Binning_ContDV


# GLOBALS
local_data_root = '/Users/vnathan/CustomerValueModelling/'
project = local_data_root+'ModelForSettingComp/WithCustomerServicing/'
raw_data = project + 'raw_data/'
feat_sel = project + 'feature_selection/'
experiment = project + 'experiments/'
segment_path = project + 'segmentation/'
inp_file = 'final_modelling_dataset_Oct172017_SC_preprocessed_c60.pkl'
segment_file = 'segmentationvars_DV1.csv'
grid_path = project + 'gridsearch/'
out_file = 'segmentation_results'
DV = 'DV1'
feats = [
 'MI_LineAsof_ObsDate', 'YL_Deposit_credit_total_61_90',
 'CR_InquiriesDuringLast6Months', 'CR_RealEstateBalance',
 'LI_YL_Cbl30median_Fewaratio_Whl3mkbgbaltobl30median_L90',
 'PP_RevenueDebitRegulCountWeekly_L90I1', 'MI_segment_BAB',
 'MI_UtilisationAsof_ObsDate', 'RV_ScorableRev30_HL360D',
 'CR_RevolvingBalance', 'MI_ADBAsof_1M_ObsDate',
 'CR_MonthsSinceOldestTradeOpen', 'CR_DelinquenciesOver60Days',
 'EA_IQC9416', 'MI_ChangeINRev30From_qual',
 'LI_YL_Cbl30min_Fmonthlyratio_Wcurrentbalancetobl30min_L90',
 'YL_CreditSkew_L90I1', 'KH_BalanceAccel', 'MI_ADB_qual',
 'RV_ScorableRev30Accel_HL90D', 'PP_DebitCategFract_RefundOffer',
 'PP_DebitCategFract_Reversal', 'MI_fee_12month',
 'PP_RevenueDebitMedian_L90I1', 'YL_DebitDailyTotalCV_L90I1',
 'LI_YL_Cbl30maxdiff_Fmonthlyratio_Wcurrentbalancetobl30maxdiff_L90',
 'YL_Transfer_debit_count_1_30', 'KH_NumPmts_L7D',
 'PPB_ModeLog10_L90I1', 'MI_ChangeInLineFrom_qual',
 'YL_Shipping_debit_total_61_90', 'PP_OtherCreditSundayPctTotal_L90I1',
 'YL_DebitCategFract_Uncategorized', 'YL_MercProc_debit_count_61_90',
 'EA_RPM8100', 'PP_RevenueCreditModePctDays_L90I1',
 'PP_OtherDebitEverydayPctUnique_L90I1', 'MI_acqavailableline_Tier3',
 'PP_CreditCategFract_FeeReversal', 'PP_RevenueCreditMean_L90I1',
 'KH_CashflowAccel_HL12E', 'EA_IQA9510', 'CR_FICOScore',
 'LI_PP_Cbl30median_Fmonthlyratio_Wlinesizetobl30median_L90',
 'LI_YL_Cbl30mediandiff_Fmonthlyratio_Wcurrentbalancetobl30mediandiff_L90',
 'YL_CreditCategFract_Loans', 'EA_BCC3476']
coef = [184.562, 66.1709, -152.8141, 186.012, -123.4066, 21.3104, 169.517,
        -116.4776, 14.926, 83.0553, 53.7358, 46.3514, 125.4496, -106.3909,
        108.8828, -130.2496, -33.5619, -17.8579, 40.2451, 29.0215, -114.0378,
        -44.3627, 194.0618, 44.4787, -63.7186, 46.7833, 36.1759, 117.2757,
        41.3126, 12.9585, -75.6178, -25.5127, -91.7927, -64.161, -10.9902,
        20.6687, 17.2448, -113.9225, -14.3451, -128.0101, -31.85, -21.4735,
        -23.1961, 197.8739, -62.1357, -36.3779, -31.5875]
coef_dict = dict(zip(feats, coef))
model_path = grid_path + 'best_smf_ols_model.pkl'


def read_datafile(path):
    fn, ext = os.path.splitext(path)
    read_fn = {'.csv': pd.read_csv,
               '.pkl': joblib.load,
               '.json': json.load}
    return read_fn.get(ext, pd.read_csv)(path)


def add_score(data, model):
    """
    data: preprocessed data
    """
    df = data.copy()
    df.reset_index(drop=True, inplace=True)
    df['Score'] = model.predict(df[feats])
    return df


def add_additional_cols(data):
    df = data.copy()
    df['SquaredError_DS'] = map(lambda x, y: (x-y)**2, df[DV], df['Score'])
    # add MKT_predicted
    mask = df['sample'] == 'dev'
    dev = df.loc[mask, :]
    dev.reset_index(drop=True, inplace=True)
    table = pd.DataFrame(dev.groupby('MI_segment')[DV].mean())
    table.reset_index(inplace=True)
    map_dict = dict(zip(table['MI_segment'], table[DV]))
    df['MKT_predicted'] = df['MI_segment'].map(map_dict)
    df['SquaredError_MKT'] = map(lambda x, y: (x-y)**2, df[DV],
                                 df['MKT_predicted'])
    return df


def get_binned_scores(data, depth):
    """
    bins scores from regression model using decision tree
    data: dataframe with shape (nsamples, nfeatures)
    """
    df = data.copy()
    binning = Binning_ContDV(data=df, feat_col='Score', index='ak_concat',
                             sample_col='sample', target=DV, max_depth=depth)
    df_binned = binning.get_binned_data()

    return df_binned


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


def display(data, sample='val', bins='Score_binned'):
    """
    data: DataFrame with shape (nsamples, nfeatures)
    bins: {'DS', 'MKT'}
    sample: value for which table should be displayed
    """
    df = data[data['sample'] == sample]
    df.reset_index(drop=True, inplace=True)
    if bins == 'Score_binned':
        pred = 'Score'
        deviance = 'SquaredError_DS'
    else:
        pred = 'MKT_predicted'
        deviance = 'SquaredError_MKT'

    agg_pred = {pred: [np.min, np.max, np.nanmean, np.nanmedian]}
    agg_num_accs = {'ak_concat': len}
    agg_actual_12 = {DV: [np.nanmean, np.nanmedian, perc10, perc90]}
    agg_deviance = {deviance: rmse}
    agg_line = {'availableline': [np.nanmean, np.sum]}
    agg_line_period = {'availableline_6month': np.nanmean,
                       'availableline_12month': np.nanmean,
                       'availableline_24month': np.nanmean}
    agg_price = {'Fee_6month': np.nanmean, 'Fee_12month': np.nanmean}
    agg_others = {'ficoscore': np.nanmean, 'ficoscore_6month': np.nanmean,
                  'ficoscore_12month': np.nanmean, 'ficoscore_24month': np.nanmean,
                  'utilisation': np.nanmean, 'utilisation_6month': np.nanmean,
                  'utilisation_12month': np.nanmean,
                  'utilisation_24month': np.nanmean,
                  'totaladvances': np.sum,
                  'totaladvances_6month': np.sum,
                  'totaladvances_12month': np.sum,
                  'totaladvances_24month': np.sum,
                  'totaladvancecount': np.sum,
                  'totaladvancecount_6month': np.sum,
                  'totaladvancecount_12month': np.sum,
                  'totaladvancecount_24month': np.sum,
                  'DelinquentOrCOFlag_12month': np.nanmean,
                  'DelinquentOrCOFlag_24month': np.nanmean,
                  'rev30': np.nanmean, 'rev30_6month': np.nanmean,
                  'rev30_12month': np.nanmean, 'rev30_24month': np.nanmean,
                  'adb': np.nanmean, 'adb_6month': np.nanmean,
                  'adb_12month': np.nanmean, 'adb_24month': np.nanmean}

    table_pred = group(df, bins, agg_pred)
    table_pred.columns = ['DS_Pred_Bins', 'Min_Pred', 'Max_Pred', 'Mean_Pred',
                          'Median_Pred']

    table_num_accs = group(df, bins, agg_num_accs)
    table_num_accs.columns = ['DS_Pred_Bins', '#Accounts']

    table_actual_12 = group(df, bins, agg_actual_12)
    table_actual_12.columns = ['DS_Pred_Bins', 'Mean_Actual',
                               'Median_Actual',
                               'Perc10_Actual', 'perc90_Actual']

    table_deviance = group(df, bins, agg_deviance)
    table_deviance.columns = ['DS_Pred_Bins', 'RMSE']

    table_line = group(df, bins, agg_line)
    table_line.columns = ['DS_Pred_Bins', 'AvgLine', 'Exposure']

    table_line_period = group(df, bins, agg_line_period)
    table_line_period.rename(columns={
     bins: 'DS_Pred_Bins', 'availableline_6month': 'AvgLine_6month',
     'availableline_12month': 'AvgLine_12month',
     'availableline_24month': 'AvgLine_24month'}, inplace=True)

    table_price = group(df, bins, agg_price)
    table_price.rename(columns={bins: 'DS_Pred_Bins'}, inplace=True)

    table_others = group(df, bins, agg_others)
    table_others.rename(columns={
     bins: 'DS_Pred_Bins', 'ficoscore': 'AvgFICO',
     'ficoscore_6month': 'AvgFICO_6month',
     'ficoscore_12month': 'AvgFICO_12month',
     'ficoscore_24month': 'AvgFICO_24month',
     'totaladvances': 'TotalAdvances@Obs',
     'totaladvances_6month': 'TotalAdvances_6month',
     'totaladvances_12month': 'TotalAdvances_12month',
     'totaladvances_24month': 'TotalAdvances_24month',
     'totaladvancecount': 'TotalNumAdvances@Obs',
     'totaladvancecount_6month': 'TotalNumAdvances_6month',
     'totaladvancecount_12month': 'TotalNumAdvances_12month',
     'totaladvancecount_24month': 'TotalNumAdvances_24month',
     'DelinquentOrCOFlag_12month': 'PercDelinquentCO_12months',
     'DelinquentOrCOFlag_24month': 'PercDelinquentCO_24months',
     'utilisation': 'utilisation@Obs'
                                 }, inplace=True)

    # combine
    table_num_accs.drop('DS_Pred_Bins', axis=1, inplace=True)
    table_actual_12.drop('DS_Pred_Bins', axis=1, inplace=True)
    table_deviance.drop('DS_Pred_Bins', axis=1, inplace=True)
    table_line.drop('DS_Pred_Bins', axis=1, inplace=True)
    table_line_period.drop('DS_Pred_Bins', axis=1, inplace=True)
    table_price.drop('DS_Pred_Bins', axis=1, inplace=True)
    table_others.drop('DS_Pred_Bins', axis=1, inplace=True)
    table = pd.concat([table_pred, table_num_accs, table_actual_12,
                       table_line_period,
                       table_deviance, table_line, table_price, table_others],
                      axis=1)
    cols = ['DS_Pred_Bins', 'Min_Pred', 'Max_Pred', 'Mean_Pred', 'Median_Pred',
            '#Accounts', 'Mean_Actual', 'Median_Actual',
            'Perc10_Actual', 'perc90_Actual', 'RMSE', 'AvgLine',
            'AvgLine_6month', 'AvgLine_12month', 'AvgLine_24month', 'Exposure',
            'Fee_6month', 'Fee_12month', 'AvgFICO', 'AvgFICO_6month',
            'AvgFICO_12month', 'AvgFICO_24month', 'utilisation@Obs',
            'utilisation_6month', 'utilisation_12month', 'utilisation_24month',
            'TotalAdvances@Obs', 'TotalAdvances_6month',
            'TotalAdvances_12month', 'TotalAdvances_24month',
            'TotalNumAdvances@Obs',
            'TotalNumAdvances_6month', 'TotalNumAdvances_12month',
            'TotalNumAdvances_24month', 'PercDelinquentCO_12months',
            'PercDelinquentCO_24months', 'rev30', 'rev30_6month',
            'rev30_12month', 'rev30_24month', 'adb', 'adb_6month',
            'adb_12month', 'adb_24month']

    total = ['Total', df[pred].min(), df[pred].max(), df[pred].mean(),
             df[pred].median(), df.shape[0], df[DV].mean(),
             df[DV].median(), perc10(df[DV]), perc90(df[DV]),
             rmse(df[deviance]),
             df['availableline'].mean(),
             df['availableline_6month'].mean(),
             df['availableline_12month'].mean(),
             df['availableline_24month'].mean(),
             df['availableline'].sum(),
             df['Fee_6month'].mean(),
             df['Fee_12month'].mean(), df['ficoscore'].mean(),
             df['ficoscore_6month'].mean(), df['ficoscore_12month'].mean(),
             df['ficoscore_24month'].mean(),
             df['utilisation'].mean(), df['utilisation_6month'].mean(),
             df['utilisation_12month'].mean(),
             df['utilisation_24month'].mean(),
             df['totaladvances'].sum(),
             df['totaladvances_6month'].sum(),
             df['totaladvances_12month'].sum(),
             df['totaladvances_24month'].sum(),
             df['totaladvancecount'].sum(),
             df['totaladvancecount_6month'].sum(),
             df['totaladvancecount_12month'].sum(),
             df['totaladvancecount_24month'].sum(),
             df['DelinquentOrCOFlag_12month'].mean(),
             df['DelinquentOrCOFlag_24month'].mean(),
             df['rev30'].mean(), df['rev30_6month'].mean(),
             df['rev30_12month'].mean(), df['rev30_24month'].mean(),
             df['adb'].mean(), df['adb_6month'].mean(),
             df['adb_12month'].mean(), df['adb_24month'].mean()]
    table = table[cols]
    table.loc[table.shape[0]] = total

    return table


def crosstab(data, sample='val'):
    df = data[data['sample'] == sample]
    df.reset_index(drop=True, inplace=True)
    table = pd.crosstab(df['Score_binned'], df['MI_segment'], margins=True)
    table.reset_index(inplace=True)
    table.rename(columns={'Score_binned': 'SegmentMatrix'}, inplace=True)
    return table


if __name__ == '__main__':
    print 'segmentation is running...'
    print 'read data'
    df = read_datafile(feat_sel+inp_file)
    model = read_datafile(model_path)
    df1 = add_score(df, model)
    df2 = read_datafile(segment_path+segment_file)
    cols = ['sample', 'Score', 'DV1']
    df2.drop(cols, axis=1, inplace=True)
    data = pd.merge(df2, df1[cols+['ak_concat']], on='ak_concat')
    print 'adding additional cols'
    data = add_additional_cols(data)
    print 'segmentation'
    data1 = get_binned_scores(data=data, depth=3)
    data1.to_csv(segment_path+'final_data_segmentation.csv', index=False)
    print 'display function is starting...'
    print 'val sample + DS Solution:'
    table_val_ds = display(data1, sample='val', bins='Score_binned')
    table_val_ds.to_csv(segment_path+out_file+'_'+DV+'_val_ds.csv', index=False)

    print 'val sample + MKT Solution:'
    table_val_mkt = display(data1, sample='val', bins='MI_segment')
    table_val_mkt.to_csv(segment_path+out_file+'_'+DV+'_val_mkt.csv', index=False)

    print 'crosstab:'

    print 'val sample:'
    table_val = crosstab(data1, sample='val')
    table_val.to_csv(segment_path+'crosstab_val_'+DV+'.csv', index=False)

    # save binned scores table
    cols = ['ak_concat', 'Score_binned']
    raw = read_datafile(raw_data+'final_modelling_dataset_Oct172017_SC.csv')
    raw1 = pd.merge(raw, data1[cols], on='ak_concat')
    for i in raw1['Score_binned'].unique().tolist():
        name = 'final_modelling_dataset_Oct172017_SC_segment{}.csv'.format(i+1)
        mask = raw1['Score_binned'] == i
        out = raw1.loc[mask, :]
        out.reset_index(drop=True, inplace=True)
        out.to_csv(raw_data+name, index=False)
