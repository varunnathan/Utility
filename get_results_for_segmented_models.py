
model = '35f6e71aR'
df = seg5.copy()
mask = df['sample'] == 'dev'
mean_dev_actual = df.loc[mask, 'DV1'].mean()
df1 = pd.merge(df, tmp[['MKT_segment', 'ak_concat']], on='ak_concat')
dev = df1[df1['sample'] == 'dev']
dev.reset_index(drop=True, inplace=True)
table = pd.DataFrame(dev.groupby('MKT_segment')['DV1'].mean())
table.reset_index(inplace=True)
val_dict = dict(zip(table['MKT_segment'], table['DV1']))
df1['MKT_predicted'] = df1['MKT_segment'].map(val_dict)
mask = df1['sample'] == 'dev'
df1['mean_dev_actual'] = df1.loc[mask, 'DV1'].mean()
print calc_rmse(df1.loc[mask, 'DV1'], df1.loc[mask, 'MKT_predicted']) # dev
print calc_rmse(df1.loc[~mask, 'DV1'], df1.loc[~mask, 'MKT_predicted']) # val
print calc_rmse(df1.loc[mask, 'DV1'], df1.loc[mask, 'mean_dev_actual']) # dev
print calc_rmse(df1.loc[~mask, 'DV1'], df1.loc[~mask, 'mean_dev_actual']) # val

print r2_score(df1.loc[mask, 'DV1'], df1.loc[mask, 'MKT_predicted']) # dev
print r2_score(df1.loc[~mask, 'DV1'], df1.loc[~mask, 'MKT_predicted']) # val
print r2_score(df1.loc[mask, 'DV1'], df1.loc[mask, 'mean_dev_actual']) # dev
print r2_score(df1.loc[~mask, 'DV1'], df1.loc[~mask, 'mean_dev_actual']) # val

df1.loc[mask, 'DV1'].describe() # dev
df1.loc[~mask, 'DV1'].describe() # val

score = joblib.load('/Users/vnathan/.qapla/models/'+model+'/scores.pkl')
mask_score = score['sample'] == 'dev'
print calc_rmse(score.loc[mask_score, 'DV1'], score.loc[mask_score, 'Score']) # dev
print calc_rmse(score.loc[~mask_score, 'DV1'], score.loc[~mask_score, 'Score']) # val
print r2_score(score.loc[mask_score, 'DV1'], score.loc[mask_score, 'Score'])
print r2_score(score.loc[~mask_score, 'DV1'], score.loc[~mask_score, 'Score'])
