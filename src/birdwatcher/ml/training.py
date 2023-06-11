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
    y_cols = "sentiment140_target"
    x_cols = list(df_sentiment140_tfidf.columns)
    x_cols.remove(y_cols)
    x_tfidf = df_sentiment140_tfidf[x_cols]
    y_tfidf = df_sentiment140_tfidf[y_cols]
    x_train, x_test, y_train, y_test = train_test_split(
        x_tfidf, 
        y_tfidf, 
        test_size=TRAINING_CONFIG.test_size,
        random_state=TRAINING_CONFIG.random_state
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
    random_state: int = TRAINING_CONFIG.random_state,
    save: bool = True,
    verbose: bool = True
):
    """
    Get inference pipeline object.
    """
    _logger.info(
        "Defining pre-training pipeline."
    )
    # Combine dataprep, feature generation and trainteset into a
    # pre-training pipeline.
    dataprep_pipeline = get_dataprep_pipeline(
        dataprep_kwargs=dataprep_kwargs
    )
    feature_generation_pipeline = get_feature_generation_pipeline(
        tfidf_kwargs=tfidf_kwargs
    )
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

    # Load raw sentiment140 and fit pre-training pipeline.
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
    
    # Fit model pipeline.
    _logger.info(
        "Fitting model pipeline."
    )
    model_pipeline = Pipeline(
        steps=[("pca", pca), ("classifier", model)], verbose=verbose
    )
    model_pipeline.fit(X=x_train, y=y_train)
    
    # Define trained inference pipeline.
    trained_training_pipeline = Pipeline(
        steps=[("dataprep", dataprep_pipeline),
               ("feature_generation", feature_generation_pipeline),
               ("traintest", traintest),
               ("pca", model_pipeline["pca"]),
               ("classifier", model_pipeline["classifier"])],
        verbose=verbose
    )

    # Save the results, if applicable
    if save:
        _logger.info(
            "Saving training pipeline to "
            f"{PATHS.training_pipeline_path}."
        )
        with open(PATHS.training_pipeline_path, "wb") as outfile:
            pickle.dump(trained_training_pipeline, outfile)
        _logger.info(
            f"Saving trained pca to {PATHS.pca_path}."
        )
        with open(PATHS.pca_path, "wb") as outfile:
            pickle.dump(trained_training_pipeline["pca"], outfile)
        _logger.info(
            f"Saving trained model to {PATHS.model_path}."
        )
        with open(PATHS.model_path, "wb") as outfile:
            pickle.dump(trained_training_pipeline["classifier"], outfile)
    return trained_training_pipeline
