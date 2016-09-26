"""
Description: Class ProfileSpark
"""

from __future__ import division, print_function

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pyspark.sql import functions as f


class ProfileSparkDF:
    def __init__(self, df, ctx):
        """
        Simple profiling of the data in a Spark DataFrame

        Parameters
        ----------
        df : pyspark.sql.dataframe.DataFrame, Spark DataFrame to profile
        ctx : pyspark.sql.context.HiveContext, HiveContext of the Spark Context

        Examples
        --------
        # >>> profiler = ProfileSparkDF(df, sqlContext)
            **** Column = Fare
            type                         num
            dType                     double
            missing_data_fraction          0
            distinct_values              250
            zeros_fraction             1.684
            avg                      32.2042
            stddev_of_avg            1.66479
            stddev                   49.6934
            min                            0
            pc25                     7.89575
            pct50                     14.4426
            pc75                     30.9239
            max                      512.329
            Name: Fare, dtype: object
            ...

        # >>> profiler.df_describe_data
                    type   dType  missing_data_fraction ...
        Name         cat  string                 99.888
        Age          num  double                 80.135
        Pclass       num  double                 99.888
        Sex          cat  string                 99.888
        Survived     num  bigint                100.000
        PassengerId  num  bigint                100.000

        # >>> profiler.null_data_columns
        ['NullColumn']

        """
        # initialize
        self.df = df
        self.ctx = ctx
        self._df_columns = self.df.columns
        self._df_count = self.df.count()

        # cat and num vars
        self._df_columns_dtypes = dict(self.df.dtypes)
        self._df_columns_type = dict()
        self._columns_types_defs_dict = {'bigint': 'num', 'int': 'num', 'double': 'num', 'float': 'num',
                                         'binary': 'cat', 'bin': 'cat', 'string': 'cat'}
        for col, dtype in self._df_columns_dtypes.items():
            self._df_columns_type[col] = self._columns_types_defs_dict[dtype]

        # list of vars to use for describe data.
        # NOTE: this list defines the column order of df_describe_data
        self._columns_describe_vars = [
            'type',
            'dType',
            'missing_data_fraction',
            'distinct_values',
            'zeros_fraction',
            'avg',
            'stddev_of_avg',
            'stddev',
            'min',
            'pc25',
            'pc50',
            'pc75',
            'max',
        ]
        # save output as dict
        # {col: {var: value}}
        self._columns_describe_dict = \
            dict([(col, dict([(s, None) for s in self._columns_describe_vars])) for col in self._df_columns])
        # and as pandas df
        self.df_describe_data = pd.DataFrame(data=self._columns_describe_dict.values(),
                                             index=self._columns_describe_dict.keys())
        # set columns order in the df
        self.df_describe_data = self.df_describe_data[self._columns_describe_vars]

        # save data for plotting charts
        self._frequencies_dict = dict()
        self._distributions_dict = dict()

        # set
        self.f_max = 25  # for plotting frequency bar char. Limit the chart to show f_max most frequent values
        self.n_bins = 25  # for distribution histo

        # list of null columns
        self.null_data_columns = list()

        # spark df with not null data per each column.
        # {col: (df.cached(), df_count)}
        self.not_null_data_dict = dict()
        for col, dtype in self._df_columns_type.items():
            df, df_count = self._get_col_not_null_values(col, dtype)
            # if column is not null
            if df_count != 0:
                self.not_null_data_dict[col] = (df.cache(), df_count)
            else:
                self.null_data_columns.append(col)

        # analyse if dataframe is not empty
        if self._df_count != 0:

            self._save_col_types()

            for column in self.not_null_data_dict.keys():
                # analyze
                self._calc_null_distinct(column)
                self._calc_num_stats(column)
                self._calc_frequencies_and_distributions(column, f_max=self.f_max, n_bins=self.n_bins)
                self.df_describe_data = pd.DataFrame(data=self._columns_describe_dict.values(),
                                                     index=self._columns_describe_dict.keys())
                self._update_df_describe_data()
                # print result
                self._print_column_data_profile(column)
        else:
            print("\n****INFO:: EMPTY DataFrame\n")

    def _get_col_not_null_values(self, column, col_type):
        """
        Get only null data and return
        Returns (df, df_count)

        Notes:
            df = spark df with one column (aka series) with non null values only
            df_count = number of rows of spark df
        """
        df = self.df.select(column)
        # better way check string type empty?
        df1 = df.where(f.col(column) != '') if col_type == 'cat' else df.dropna()
        return df1, df1.count()

    def _save_col_types(self):
        """
        Save col types in output df
        """
        for col in self._df_columns:
            c_dict = self._columns_describe_dict[col]
            c_dict['type'] = self._df_columns_type[col]
            c_dict['dType'] = self._df_columns_dtypes[col]

    def _calc_null_distinct(self, column):
        """
        Calc the fraction of non null data and number of distinct valued for given column
        """
        c_dict = self._columns_describe_dict[column]
        df = self.not_null_data_dict[column][0]
        df_count = self.not_null_data_dict[column][1]
        c_dict['missing_data_fraction'] = round(100 - df_count / self._df_count * 100, 3)
        # approxCountDistinct or countDistinct?
        c_dict['distinct_values'] = df.agg(f.approxCountDistinct(df[column])).first()[0]

    def _calc_num_stats(self, column):
        """
        Calc fraction simples statistical metrics

        Notes:
            - stddev_of_avg = std dev of the average estimator
            - pc25 = First percentile, 25% of population is below pc25
        """
        agg_f = {'avg': f.avg, 'max': f.max, 'min': f.min, 'stddev': f.stddev}
        # exit if column not numbers
        if self._df_columns_type[column] != 'num':
            return

        c_dict = self._columns_describe_dict[column]
        df = self.not_null_data_dict[column][0]
        df_count = self.not_null_data_dict[column][1]

        for f_name, func in agg_f.items():
            c_dict[f_name] = df.agg(func(df[column])).first()[0]

        c_dict['stddev_of_avg'] = c_dict['stddev'] / np.sqrt(df_count)

        # define desired percentile
        percentiles_def_dict = {'pc25': .25, 'pc50': .50, 'pc75': .75}
        percentiles_values_dict = self._generate_percentile(df, column, percentiles_def_dict)
        for k in percentiles_def_dict.keys():
            c_dict[k] = percentiles_values_dict[k]

        c_dict['zeros_fraction'] = round(df.where(df[column] == 0).count() / self._df_count * 100, 3)

    def _calc_frequencies_and_distributions(self, column, f_max, n_bins):
        """
        Calc the value frequencies and histograms
        """
        # df for the column
        df = self.not_null_data_dict[column][0]

        self._frequencies_dict[column] = df.groupby(column).count().orderBy('count', ascending=False).limit(
            f_max).toPandas()

        if self._df_columns_type[column] == 'num':
            self._distributions_dict[column] = generate_histogram(df, n_bins=n_bins, col=column)  # returns spark df

    def _generate_percentile(self, df, col, percentiles_def_dict):
        """
        Calc percentiles for a column

        Args:
            df:
            col:
            percentiles_def_dict: e.g. {'pc25': .25, 'pc50': .50, 'pc75': .75}

        Returns:
            percentiles_values_dict: e.g. {'pc25': 7.89575, 'pc50': 14.4426, 'pc75': 30.9239}
        """
        self.ctx.registerDataFrameAsTable(df, "df")

        percentiles_query_dict_values = \
            self.ctx.sql("SELECT {0} FROM df".format(', '.join(
                "percentile_approx(" + col + ",%r) as %s" % (val, key) for (key, val) in
                percentiles_def_dict.items()))).collect()

        percentiles_values_dict = dict.fromkeys(percentiles_def_dict.keys())
        for k, v in percentiles_def_dict.items():
            percentiles_values_dict[k] = percentiles_query_dict_values[0][k]

        self.ctx.dropTempTable('df')

        return percentiles_values_dict

    def _update_df_describe_data(self):
        """
        Updates the df_describe_data with results
        """
        self.df_describe_data = pd.DataFrame(data=self._columns_describe_dict.values(),
                                             index=self._columns_describe_dict.keys())
        # set order
        self.df_describe_data = self.df_describe_data[self._columns_describe_vars]

    def _print_column_data_profile(self, column):
        """
        Returns Human readable profiling results for one column
        """
        print('\n**** Column = ' + column)
        print(self.df_describe_data.loc[column])

        # print Charts for non null columns
        try:
            self.not_null_data_dict[column]
        except KeyError:
            print('****Column All Null')
        else:
            df_frequency = self._frequencies_dict[column]
            df_frequency.sort_values(by='count', inplace=True)
            df_frequency.plot(kind='barh', x=column, y='count', title='Frequency of ' + column)
            plt.show()
            if self._df_columns_type[column] == 'num':
                df_distribution = self._distributions_dict[column]
                plt.hist(df_distribution['bin'], weights=df_distribution['count'])
                plt.title('Distribution of ' + column)
                plt.show()


def generate_histogram(df, n_bins, col):
    """
    Returns histogram as pandas df , of the format:
             bin  count
    0   3.494000   55.0
    1  11.350000   20.0
    2  19.359649  171.0
    ...
    """
    df_histo = df.selectExpr('histogram_numeric({0}, {1})'.format(col, n_bins))  # returns spark DF
    return pd.DataFrame([(r.x, r.y) for r in df_histo.first()[0]], columns=['bin', 'count'])
