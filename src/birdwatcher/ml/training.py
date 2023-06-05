"""
Train sentiment analysis model.
"""


from dataclasses import dataclass, field
import logging
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from typing import Optional, Union


from birdwatcher.config import DATA_PREP_CONFIG, PATHS, TRAINING_CONFIG
from birdwatcher.dataprep.prep import get_dataprep_pipeline
from birdwatcher.ml.feature_generation import (
    _get_tfidf_save_path,
    get_feature_generation_pipeline
)
from birdwatcher.ml.PCAPlotIt import PCAPlotIt


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_formatter = logging.Formatter(
    "%(asctime)s:%(levelname)s:%(module)s: %(message)s"
)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)
_logger.addHandler(_console_handler)


TRAINING_DATAPREP_KWARGS = {
    "text_col": "text",
    "data_key": "sentiment140",
    "save_path": DATA_PREP_CONFIG.processing_info["sentiment140"]["path_proc"]
}
TRAINING_TFIDF_KWARGS = {
    "text_col_proc": "text_processed",
    "data_key": "sentiment140",
    "save": True
}


@dataclass
class TrainTestSplit():

    """
    Store train/test split.
    """

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: Union[pd.DataFrame, pd.Series]
    y_test: Union[pd.DataFrame, pd.Series]


def _get_train_test_data(
    df_sentiment140_tfidf: Optional[pd.DataFrame] = None
) -> TrainTestSplit:
    """
    Get train and test data.
    """
    if df_sentiment140_tfidf is None:
        tfidf_path = _get_tfidf_save_path(data_key="sentiment140")
        df_sentiment140_tfidf = pd.read_parquet(tfidf_path)
    y_cols = "target"
    x_cols = list(df_sentiment140_tfidf.columns)
    x_cols.remove(y_cols)
    x_tfidf = df_sentiment140_tfidf[x_cols]
    y_tfidf = df_sentiment140_tfidf[y_cols]
    x_train, x_test, y_train, y_test = train_test_split(
        x_tfidf, 
        y_tfidf, 
        test_size=TRAINING_CONFIG.train_test_info["test_size"],
        random_state=TRAINING_CONFIG.train_test_info["random_state"]
    )
    return TrainTestSplit(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test
    )


def _get_training_data(
    df_sentiment140_tfidf: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Get x_train data.
    """
    train_test_split = _get_train_test_data(
        df_sentiment140_tfidf=df_sentiment140_tfidf
    )
    return (train_test_split.x_train, train_test_split.y_train)


def train_save_inference_pipeline(
    dataprep_kwargs: dict = TRAINING_DATAPREP_KWARGS,
    tfidf_kwargs: dict = TRAINING_TFIDF_KWARGS,
    pca_kwargs: dict = TRAINING_CONFIG.pca_kwargs,
    model_params: dict = TRAINING_CONFIG.model_params,
    save: bool = True,
    verbose: bool = True
):
    """
    Get inference pipeline object.
    """
    _logger.info(
        "Defining pre-training pipeline."
    )
    # Get dataprep and feature generation pipelines
    dataprep_pipeline = get_dataprep_pipeline(
        dataprep_kwargs=dataprep_kwargs
    )
    feature_generation_pipeline = get_feature_generation_pipeline(
        tfidf_kwargs=tfidf_kwargs
    )
    
    # Get train/test split, pca and model
    traintest = FunctionTransformer(
        _get_training_data,
        validate=False
    )
    pretrain_pipeline = Pipeline(
        steps=[("dataprep", dataprep_pipeline),
               ("feature_generation", feature_generation_pipeline),
               ("traintest", traintest)],
        verbose=verbose
    )

    _logger.info(
        "Loading raw sentiment140 dataset."
    )
    df_sentiment_raw = pd.read_parquet(
        DATA_PREP_CONFIG.processing_info["sentiment140"]["path_raw"]
    )

    _logger.info(
        "Fitting pre-training pipeline and transforming raw dataset."
    )
    x_train, y_train = pretrain_pipeline.fit_transform(X=df_sentiment_raw)

    pca = PCAPlotIt(**pca_kwargs)
    model = LogisticRegression(**model_params)

    model_pipeline = Pipeline(
        steps=[("pca", pca), ("classifier", model)], verbose=verbose
    )
    model_pipeline.fit(X=x_train, y=y_train)

    trained_inference_pipeline = Pipeline(
        steps=[("dataprep", dataprep_pipeline),
               ("feature_generation", feature_generation_pipeline),
               ("pca", model_pipeline.named_steps["pca"]),
               ("classifier", model_pipeline.named_steps["classifier"])],
        verbose=verbose
    )

    # Save the results, if applicable
    if save:
        _logger.info(
            "Saving fitted inference pipeline to "
            f"{PATHS.inference_pipeline_path}."
        )
        with open(PATHS.inference_pipeline_path, "wb") as outfile:
            pickle.dump(trained_inference_pipeline, outfile)
    return trained_inference_pipeline




# def get_pca(df_sentiment140_tfidf_train: Optional[pd.DataFrame] = None):
#     """
#     Get a fitted PCAPlotIt object.
#     """
#     # Fit a PCAPlotIt object, if one doesn't already exist.
#     if not os.path.exists(PATHS.pca_path):
#         _logger.info(
#             "Couldn't find a saved PCAPlotIt. "
#             "Instantiating new object with the following kwargs: "
#             f"{TRAINING_CONFIG.pca_kwargs}."
#         )
#         pca = PCAPlotIt(
#             **TRAINING_CONFIG.pca_kwargs
#         )
#         if df_sentiment140_tfidf_train is None:
#             tfidf_path = _get_tfidf_save_path(data_key="sentiment140")
#             df_sentiment140_tfidf = pd.read_parquet(tfidf_path)
#             x_train = _get_train_test_data(
#                 df_sentiment140_tfidf=df_sentiment140_tfidf
#             ).x_train
#         pca.fit(x_train)

#         # Save, if applicable
#         if save:
#             _logger.info(
#                 "Saving fitted PCAPlotIt object to "
#                 f"{PATHS.pca_path}."
#             )
#             with open(PATHS.pca_path, "wb") as outfile:
#                 pickle.dump(pca, outfile)
#     else:
#         _logger.info(
#             f"Found a saved PCAPlotIt. Loading from {PATHS.pca_path}."
#         )
#         with open(PATHS.pca_path, "rb") as infile:
#             pca = pickle.load(infile)
#     return pca
