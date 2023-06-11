"""
Perform model inference.
"""


import logging
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from typing import Union


from birdwatcher.config import DATA_PREP_CONFIG, PATHS
from birdwatcher.dataprep.prep import (
    _generate_raw_proc_path,
    get_dataprep_pipeline
)
from birdwatcher.ml.feature_generation import get_feature_generation_pipeline



_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_formatter = logging.Formatter(
    "%(asctime)s:%(levelname)s:%(module)s: %(message)s"
)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)
_logger.addHandler(_console_handler)


with open(PATHS.dc_run_date_info, "rb") as infile:
    run_info = pickle.load(infile)


def _generate_inference_dataprep_kwargs(
    data_key: str,
    end_date_name: str = run_info.end_date_name,
    save: bool = True
):
    """
    Generate kwargs for dataprep inference pipeline.
    """
    config_dict = DATA_PREP_CONFIG.processing_info[data_key]
    kwargs = {
        "data_key": data_key,
        "text_col": config_dict["text_col"]
    }
    if save:
        if data_key == "sentiment140":
            kwargs["save_path"] = (
                DATA_PREP_CONFIG.processing_info["sentiment140"]["path_proc"]
            )
        else:
            df_proc_save_path = _generate_raw_proc_path(
                config_dict=config_dict,
                df_type="proc",
                end_date_name=end_date_name
            )
            kwargs["save_path"] = df_proc_save_path
    return kwargs


def _generate_inference_tfidf_kwargs(
    data_key: str,
    save: bool = True
):
    """
    Generate kwargs for tfidf inference pipeline.
    """
    config_dict = DATA_PREP_CONFIG.processing_info[data_key]
    kwargs = {
        "data_key": data_key,
        "text_col_proc": config_dict["text_col"] + "_processed",
        "save": save
    }
    return kwargs


def _remove_cols(
    df: pd.DataFrame,
    labels: Union[str, list[str]]
) -> pd.DataFrame:
    """
    Remove col from dataframe
    """
    return df.drop(labels=labels, axis=1)


def _get_scores_save_path(
    data_key: str,
    end_date_name: str = run_info.end_date_name
):
    """
    Get save path for scores.
    """
    fname = "scores_" + data_key + f"_{end_date_name}.parquet"
    return PATHS.scores_path + f"/{data_key}/{fname}"


def get_trained_inference_pipeline(
    data_key: str,
    end_date_name: str = run_info.end_date_name,
    save: bool = True,
    verbose: bool = True
):
    """
    Load trained inference pipeline.
    """
    # Generate dataprep and tfidf kwarg & corresponding pipelines.
    _logger.info(
        "Getting dataprep and feature generation pipelines."
    )
    INFERENCE_DATAPREP_KWARGS = _generate_inference_dataprep_kwargs(
        data_key=data_key,
        end_date_name=end_date_name,
        save=save
    )
    INFERENCE_TFIDF_KWARGS = _generate_inference_tfidf_kwargs(
        data_key=data_key,
        save=save
    )
    dataprep_pipeline = get_dataprep_pipeline(
        dataprep_kwargs=INFERENCE_DATAPREP_KWARGS
    )
    feature_generation_pipeline = get_feature_generation_pipeline(
        tfidf_kwargs=INFERENCE_TFIDF_KWARGS
    )

    # Load trained pca & model.
    _logger.info(
        f"Loading trained pca from {PATHS.pca_path}."
    )
    with open(PATHS.pca_path, "rb") as infile:
        pca = pickle.load(infile)
    _logger.info(
        f"Loading trained model from {PATHS.model_path}."
    )
    with open(PATHS.model_path, "rb") as infile:
        model = pickle.load(infile)
    
    # Combined components into inference pipeline.
    _logger.info(
        "Combining dataprep, feature generation, pca and model "
        "into an inference pipeline."
    )
    steps=[
        ("dataprep", dataprep_pipeline),
        ("feature_generation", feature_generation_pipeline),
        ("pca", pca),
        ("classifier", model)
    ]
    if data_key == "sentiment140":
        _logger.info(
            "Inserting 'droptarget' step to pipeline."
        )
        droptarget = FunctionTransformer(
            _remove_cols,
            kw_args={"labels": "sentiment140_target"},
            validate=False
        )
        steps.insert(2, ("droptarget", droptarget))
    inference_pipeline = Pipeline(
        steps=steps,
        verbose=verbose
    )
    return inference_pipeline


def perform_pipeline_inference(
    data_key: str,
    end_date_name: str = run_info.end_date_name,
    save: bool = True,
    verbose: bool = True
):
    """
    Retreive pipeline and perform inference.
    """
    # Retreive trained inference pipeline.
    inference_pipeline = get_trained_inference_pipeline(
        data_key=data_key,
        end_date_name=end_date_name,
        save=save,
        verbose=verbose
    )
    
    # Retreive raw data and perform inference.
    raw_path = _generate_raw_proc_path(
        config_dict=DATA_PREP_CONFIG.processing_info[data_key],
        df_type="raw"
    )
    df_raw = pd.read_parquet(raw_path)
    df_preds = pd.DataFrame({
        "preds": inference_pipeline.predict(df_raw)
    })
    if save:
        save_path = _get_scores_save_path(
            data_key=data_key,
            end_date_name=end_date_name
        )
        _logger.info(
            f"Saving scores for data_key={data_key} and date="
            f"{end_date_name} to {save_path}."
        )
        df_preds.to_parquet(save_path)
    return df_preds