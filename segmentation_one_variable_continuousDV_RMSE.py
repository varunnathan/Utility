import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import StratifiedKFold


class Binning_ContDV(BaseEstimator, TransformerMixin):
    """bins numeric variable when the DV is continuous"""
    def __init__(self, data=None, feat_col=None, index=None,
                 sample_col=None, target=None,
                 metrics='rmse', nsamples_bin=0.03, cv_folds=2,
                 max_depth=2):
        """
        data: DataFrame with shape (nsamples, nfeatures)
        feat_col: list of comma separated names of features
        index: column name of unique column
        sample_col: column name of sample_col
        target: column name of DV
        metrics: metrics used for finding the best split strategy
        {'rmse'}
        nsamples_bin: minimum percentage of samples in each bin.
        Needed if method='tree'
        cv_folds: number of cross validation folds
        max_num_clusters: parameter needed for method='cluster'
        max_depth: parameter needed for method='tree'
        """
        self.metrics = metrics
        self.nsamples_bin = nsamples_bin
        self.index = index
        self.sample = sample_col
        self.data = data
        self.target = target
        self.feature_columns = feat_col  # only one column's name
        self.cv_folds = cv_folds
        self.max_depth = max_depth

    def prepare(self):
        """
        prepares data for binning
        """
        df = self.data.copy()
        mask = df[self.sample] == 'dev'
        dev = df[mask]
        dev.reset_index(drop=True, inplace=True)
        x = dev[self.feature_columns].values
        x_overall = df[self.feature_columns].values
        self.x_overall = x_overall.reshape((x_overall.shape[0], 1))
        self.x = x.reshape((x.shape[0], 1))
        self.y = dev[self.target].values

        return self

    def apply_bins(self, X, bins):
        """
        Bins a 2d array
        X: array with shape (nsamples, 1)
        bins: 1d array of bounds
        """
        temp = pd.Series(np.copy(X).reshape(len(X),))
        binned = pd.Series([-2] * len(temp), index=temp.index)
        binned[temp.isnull()] = -1
        binned[temp < np.min(bins)] = 0

        for ibin, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
            mask = (temp >= low) & (temp < high)
            binned[mask] = ibin + 1
        binned[temp >= np.max(bins)] = len(bins)
        return binned

    def apply_bins_df(self, feature, bins):
        """
        feature: series with shape (nsamples, )
        bins: 1d array of bounds
        """
        binned = pd.Series([-2] * len(feature), index=feature.index)
        binned[feature.isnull()] = -1
        binned[(feature < np.min(bins))] = 0

        for ibin, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
            mask = (feature >= low) & (feature < high)
            binned[mask] = ibin+1
        binned[feature >= np.max(bins)] = len(bins)
        return binned

    def calc_rmse(self, y_true, y_pred):
        """rmse"""
        tmp = (y_true - y_pred)**2
        return np.sqrt(tmp.mean())

    def tree_bins(self, x, y, max_depth, n_samples_bin):
        """
        Returns bins which are based out of DecisionTreeRegressor Model.
        x: numpy array of feature col.
        y: numpy array of target_col.
        max_depth: maximum depth of tree.
        n_samples_bin: number of minimum sample in each bin.
        """

        min_samples_leaf = n_samples_bin * x.shape[0]
        model = DecisionTreeRegressor(
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf)
        model.fit(x, y)
        bins = model.tree_.threshold
        bins = np.sort(np.unique(bins))

        return model, bins

    def get_CVfolds(self, series):
        """generate  stratified folds,
         series: series with shape (nsamples, ).
        """
        folds = []
        if self.cv_folds:
            for train, test in StratifiedKFold(series, n_folds=self.cv_folds,
                                               random_state=20170322):
                if len(set(series[test])) == 1:
                    continue
                folds.append((train, test))

        if len(folds) <= 1:
            folds = [(None, None)]
        return folds

    def getargmax(self, score_dict):
        """
        returns the index of maximum value present in an array.
        score_dict : dictionary with keys as names and values as 1d array.
        """

        return np.nanargmax(score_dict["rmse_scores"])

    def tree(self, max_depth=None):
        """
        DecisionTreeRegressor
        max_depth: maximum depth of the tree. parameter used to vary the
        number of bins
        """
        m_depth = max_depth
        # get x and y for model building
        self.prepare()
        # get the folds
        folds = self.get_CVfolds(self.y)
        # performance metrics to be tracked
        params = ['rmse', 'bins_tree']
        for param in params:
            setattr(self, param, [])
        PERF_DICT = {}
        score = ['rmse_scores', 'max_depth']
        for name in score:
            PERF_DICT[name] = []
        # split the dev data into train and test
        for train, test in folds:
            modelperf_dict = {}
            for name in score:
                modelperf_dict[name + '_model'] = []
            if train is None:
                x_train, x_test, y_train, y_test = self.x, self.x, self.y, self.y
            else:
                x_train, x_test, y_train, y_test = self.x[train], self.x[test], self.y[train], self.y[test]

            # model building
            for d in range(1, m_depth):
                model, _ = self.tree_bins(x_train, y_train, d,
                                          self.nsamples_bin)
                y_test_pred = model.predict(x_test)

                # store performance metrics
                modelperf_dict["rmse_scores_model"].append(
                 self.calc_rmse(pd.Series(y_test), pd.Series(y_test_pred)))
                modelperf_dict['max_depth_model'].append(d)

                for name in score:
                    PERF_DICT[name].append(modelperf_dict[name + '_model'])

        for name in score:
            PERF_DICT[name] = np.mean(PERF_DICT[name], 0)

        index = self.getargmax(PERF_DICT)
        best_max_depth = PERF_DICT["max_depth"][index]

        """ Fit model on dev data"""
        x_, y_ = self.x, self.y
        _, bins = self.tree_bins(x_, y_, best_max_depth, self.nsamples_bin)
        self.bins_tree.append(bins)

        for pos, val in enumerate(params[:2]):
            getattr(self, val).append(PERF_DICT[score[pos]][index])

        print modelperf_dict, best_max_depth

        return self

    def get_binned_data(self):
        # get overall data
        df = self.data.copy()
        self.tree(max_depth=self.max_depth)
        df[self.feature_columns+'_binned'] = self.apply_bins(
         self.x_overall, self.bins_tree[0])
        df.reset_index(drop=True, inplace=True)
        return df
