"""
Description: Utils functions for the churn models analysis
Authors: Eliano Marques, Almir Mutapcic, Guillermo Breto Rangel, Giacomo Snidero
Created on: Aug 2016
"""

from time import time

from pyspark.sql import functions as f

from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import OneHotEncoder

from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.evaluation import MulticlassMetrics


def prepare_data(df, logger, num_cols=None, cat_cols=None, target_col=None, key_cols=None):
    """
    Steps performed on the df:
        - Cleaning
        - Filling null / empty
        - Drop non wanted columns
    """
    t_start = time()

    # cleaning
    df = df.filter("profitability!=0")
    df = df.filter("status= 'a' ")
    if "gender" in cat_cols:
        df = df[df.gender.isin("M", "F")]

    # fill na
    df = df.replace('', 'NULL', subset=cat_cols)
    ''' TODO: check if this is needed '''
    df = df.fillna(0, subset=num_cols)

    #  drop not used columns
    columns = key_cols + [target_col] + num_cols + cat_cols
    df = df.select(columns)

    logger.info("using only columns list: {}".format(columns))
    logger.info("Cleaned dataframe contains: {} rows".format(df.count()))
    logger.info("Data preparation time {} seconds".format(round(time() - t_start)))

    return df


def resample_data(df, logger, target_col, pos_to_neg_ratio=1, seed=46):
    """
    Resample df according to target_col.
    """
    t_start = time()
    target_balance_dict = df.groupby(target_col).count().rdd.collectAsMap()
    logger.info("Class balance is 1:{} 0:{}".format(target_balance_dict[1], target_balance_dict[0]))

    if pos_to_neg_ratio > 0:
        df_sampled = df.sampleBy(target_col,
                                 fractions={
                                     1: 1.0,
                                     0: float(target_balance_dict[1]) / target_balance_dict[0] / pos_to_neg_ratio},
                                 seed=seed)
        target_balance_dict = df_sampled.groupby(target_col).count().rdd.collectAsMap()
        logger.info("After Sampling the classes contain 1:{} 0:{}, pos_to_neg_ratio:{}".format(target_balance_dict[1],
                                                                                               target_balance_dict[0],
                                                                                               float(
                                                                                                   target_balance_dict[
                                                                                                       1]) /
                                                                                               target_balance_dict[0]))
        logger.info("Dataset resampling took {} seconds".format(round(time() - t_start)))
    else:
        logger.info("Dataset NOT resampled")
        df_sampled = df

    return df_sampled


def write_to_db(df, name, columns=None):
    """
    Write df to profitability db
    """
    if columns:
        df = df.select(columns)

    df.sql_ctx.sql("use profitability")
    df.write.saveAsTable(name=name, mode='overwrite')


def encode_features(df, logger, cat_features, output_col_suffix='ohe'):
    """
    Input a list of cat features
    """
    drop_last = True
    
    df_encoded = df
    cat_features_encoded_dict = {}
    for feature in cat_features:
        
        logger.info("encoding: {}".format(feature))
        
        # index
        index_name = feature + '_index'
        string_indexer = StringIndexer(inputCol=feature, outputCol=index_name)
        model_string_indexer = string_indexer.fit(df_encoded)
        df_encoded = model_string_indexer.transform(df_encoded)
        # print(model_string_indexer.labels)
        
        # encode
        feature_ohe = feature+output_col_suffix
        encoder = OneHotEncoder(dropLast=drop_last, inputCol=index_name, outputCol=feature_ohe)
        df_encoded = encoder.transform(df_encoded)
        # drop not neexed indexes
        # df_encoded = df_encoded.drop(index_name)
        # save ohe labels
        cat_features_encoded_dict[feature_ohe] = [s for s in model_string_indexer.labels]
        if drop_last:
            del cat_features_encoded_dict[feature_ohe][-1]  # delete last label

    return df_encoded, cat_features_encoded_dict


def assemble_features(df, num_features=None, cat_features=None, output_col="features"):
    """
    Assemble feature to fit into a vector object, required by spark ml
    """
    assembler = VectorAssembler(inputCols=num_features + cat_features, outputCol=output_col)
    df_assembled = assembler.transform(df)

    return df_assembled


def test_performance_cross_validation(dataset, classifier, label_col, n_folds, seed='46'):
    """
    Evaluate classifier performance using k-fold cross validation
    https://spark.apache.org/docs/1.6.0/mllib-evaluation-metrics.html
    """
    rand_col = "uid_rand"
    h = 1.0 / n_folds
    df = dataset.select("*", f.rand(seed).alias(rand_col))

    metrics_dict = {"roc_auc": [],  # roc: y=tpr x=fpr
                    "true_pos_rate": [],  # recall = true pos rate 
                    "false_pos_rate": [],
                    "precision": [],
                    "n_true_neg": [],
                    "n_false_neg": [],
                    "n_false_pos": [],
                    "n_true_pos": [], }

    model = None
    for i in range(n_folds):

        validate_lb = i * h  # lower bound
        validate_ub = (i + 1) * h  # upper bound
        condition = (df[rand_col] >= validate_lb) & (df[rand_col] < validate_ub)
        validation = df.filter(condition)
        train = df.filter(~condition)
        
        # train
        model = classifier.fit(train)
        
        # predict
        prediction = model.transform(validation)
        
        # assess performance metrics
        prediction_and_labels = prediction.rdd.map(lambda x: (x['prediction'], x[label_col]))
        print(prediction_and_labels)
        metrics = MulticlassMetrics(prediction_and_labels)
        metrics_areas = BinaryClassificationMetrics(prediction_and_labels)  # gets roc and precRecall curves
        metrics_dict['roc_auc'].append(metrics_areas.areaUnderROC)
        # a bit slow, have to calc outside loop
        cm = metrics.confusionMatrix().toArray()
        n_true_neg = cm[0, 0]
        n_false_neg = cm[1, 0]
        n_true_pos = cm[1, 1]
        n_false_pos = cm[0, 1]
        #
        metrics_dict['n_true_neg'].append(n_true_neg) 
        metrics_dict['n_false_neg'].append(n_false_neg)
        metrics_dict['n_true_pos'].append(n_true_pos)
        metrics_dict['n_false_pos'].append(n_false_pos) 
        metrics_dict['true_pos_rate'].append(n_true_pos / (n_true_pos+n_false_neg))
        metrics_dict['false_pos_rate'].append(n_false_pos / (n_false_pos+n_true_neg))
        metrics_dict['precision'].append(n_true_pos / (n_true_pos+n_false_pos))

    return model, metrics_dict
