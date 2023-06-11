"""
Purpose: Model evaluation utilities.
Author(s): Bobby Lumpkin
"""


import boto3
from collections.abc import Iterable
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.model_selection import cross_val_score


from birdwatcher.config import PATHS


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_formatter = logging.Formatter(
    "%(asctime)s:%(levelname)s:%(module)s:%(message)s"
)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)
_logger.addHandler(_console_handler)


def get_test_set_performance_metrics(
    y_test: Iterable, 
    preds_test: Iterable,
    probs_test: Iterable = None
):
    """
    Compute and log model performance metrics on a test set.
    """
    _logger.info(
        "--------------------"
        "COMPUTING TEST SET PERFORMANCE METRICS"
        "--------------------\n"
    )
    accuracy = accuracy_score(y_test, preds_test)
    _logger.info(f"Accuracy: {accuracy}")

    if probs_test is not None:
        roc_auc = roc_auc_score(y_test, probs_test)
        _logger.info(f"ROC AUC: {roc_auc}")

    conf_metrics = confusion_matrix(y_test, preds_test, labels=[0, 1])
    _logger.info("Confusion matrix:\n" + str(conf_metrics))
    _logger.info("\n" + str(classification_report(y_test, preds_test)))
    return


def get_cv_performance_metrics(
    clf,
    x_train,
    y_train,
    cv=5,
    n_jobs=-1
):
    """
    Compute and log cross-validated model performance metrics.
    """
    _logger.info(
        "--------------------"
        "COMPUTING CROSS-VALIDATED PERFORMANCE METRICS"
        "--------------------\n"
    )
    metric_list = [
        "precision",
        "recall",
        "f1_macro",
        "roc_auc"
    ]
    for metric in metric_list:
        metric_cv = np.mean(cross_val_score(
            clf, x_train, y_train, cv=cv, scoring=metric, n_jobs=n_jobs
        ))
        _logger.info(
            f"Cross-validated {metric}: {metric_cv}"
        )
    return


def get_model_performance_metrics(
    clf,
    x_train,
    y_train,
    x_test,
    y_test,
    cv=5,
    n_jobs=-1
):
    """
    Compute and log model performance metrics.
    """
    # Compute test set preds and probs.
    preds_test = clf.predict(x_test)
    probs_test = clf.predict_proba(x_test)[:,1]
    
    # Compute test set & CV metrics.
    get_test_set_performance_metrics(
        y_test=y_test,
        preds_test=preds_test,
        probs_test=probs_test
    )
    get_cv_performance_metrics(
        clf=clf,
        x_train=x_train,
        y_train=y_train,
        cv=cv,
        n_jobs=n_jobs
    )
    return

def gs_display_save_results(
    gscv_fit, 
    file_prefix: str = None,
    key: str = PATHS.dev_models,
    bucket: str = PATHS.s3_bucket.replace("s3://", ""),
    save: bool = False
):
    best_index_gs = np.argmax(gscv_fit.cv_results_["mean_test_score"])
    best_parameters_gs = gscv_fit.cv_results_["params"][best_index_gs]

    df_cv = pd.DataFrame(columns=["Params", "Mean out-of-bag F1"])
    df_cv["Params"] = gscv_fit.cv_results_["params"]
    df_cv[ "Mean out-of-bag F1"] = gscv_fit.cv_results_["mean_test_score"]
    display(df_cv.head())
    display(df_cv.tail())
    _logger.info(
        f"Best parameters: {best_parameters_gs}. Best mean out-of-bag "
        f"F1: {np.max(gscv_fit.cv_results_['mean_test_score'])}"
    )

    if save:
        if not file_prefix:
            raise ValueError("When 'save=True', 'file_prefix' can not be None.")
        best_model = gscv_fit.best_estimator_
        save_obj = {
            "df": df_cv, 
            "best_model": best_model
        }
        file_name = file_prefix + "_gscv.pickle"
        s3_save_pickle(
            save_obj, 
            key=key,  
            file_name=file_name, 
            bucket=bucket
        )
    
    return best_parameters_gs


def s3_save_pickle(
    obj, 
    key, 
    file_name=None, 
    bucket=PATHS.s3_bucket.replace("s3://", "")
):
    """
    Save an object as a pickle file to s3.
    """
    s3_resource = boto3.resource("s3")
    key = key.replace(PATHS.s3_bucket + "/", "")
    key = os.path.join(key, file_name)
    save_obj_byte = pickle.dumps(obj)
    s3_resource.Object(bucket, key).put(Body=save_obj_byte)
    return


def s3_load_pickle(
    key, 
    file_name=None, 
    bucket=PATHS.s3_bucket.replace("s3://", "")
): 
    """
    Load an object from a pickle file in s3.
    """
    s3_resource = boto3.resource("s3")
    key = key.replace(PATHS.s3_bucket + "/", "")
    key = os.path.join(key, file_name)
    response = s3_resource.Bucket(bucket).Object(key).get()
    body_string = response["Body"].read()
    obj = pickle.loads(body_string)
    return obj