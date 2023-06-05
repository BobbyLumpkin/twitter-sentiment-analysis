"""
Purpose: Generate features consumable by ml models.
Author(s): Bobby Lumpkin
"""


from dataclasses import dataclass, field
from joblib import Parallel, delayed, cpu_count
import logging
import os
import pandas as pd
from pathlib import Path
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from typing import Optional, Union


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_formatter = logging.Formatter(
    "%(asctime)s:%(levelname)s:%(module)s:%(message)s"
)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)
_logger.addHandler(_console_handler)


from birdwatcher.config import (
    DATA_PREP_CONFIG,
    FEATURE_GENERATION_CONFIG,
    PATHS
)
from birdwatcher.dataprep.prep import _load_df_if_not_provided
from birdwatcher.ml.PCAPlotIt import PCAPlotIt


with open(PATHS.dc_run_date_info, "rb") as infile:
    run_info = pickle.load(infile)


def get_tfidf_vectorizer(
    df_sentiment140_proc: Optional[pd.DataFrame] = None,
    text_col_proc: Optional[str] = None,
    save: bool = False
):
    if not os.path.exists(PATHS.tfidf_vectorizer_path):
        _logger.info(
            "Couldn't find vectorizer. "
            "Instantiating new TfidfVectorizer object with the "
            f"following kwargs: {FEATURE_GENERATION_CONFIG.vectorizer_kwargs}."
        )
        vectorizer = TfidfVectorizer(
            **FEATURE_GENERATION_CONFIG.vectorizer_kwargs
        )
        df_sentiment140_proc = _load_df_if_not_provided(
            config_dict=DATA_PREP_CONFIG.processing_info["sentiment140"],
            df=df_sentiment140_proc,
            df_type="proc",
            data_key="sentiment140",
            end_date_name=run_info.end_date_name
        )
        _logger.info(
            "Fitting Tfidfvectorizer to processed sentiment140 dataset."
        )
        vectorizer = vectorizer.fit(df_sentiment140_proc[text_col_proc].values)
        if save:
            _logger.info(
                f"Saving tfidf vectorizer to {PATHS.tfidf_vectorizer_path}."
            )
            with open(PATHS.tfidf_vectorizer_path, "wb") as outfile:
                pickle.dump(vectorizer, outfile)
    else:
        _logger.info(
            f"Found vectorizer. Loading from {PATHS.tfidf_vectorizer_path}."
        )
        with open(PATHS.tfidf_vectorizer_path, "rb") as infile:
            vectorizer = pickle.load(infile)
    return vectorizer


def _get_tfidf_save_path(
    data_key: str,
    end_date_name: str = run_info.end_date_name
) -> str:
    """
    Create the save path for a tfidf dataframe.
    """
    tfidf_dict = FEATURE_GENERATION_CONFIG.data_keys_dict[data_key]
    if data_key == "sentiment140":
        save_path=tfidf_dict["path_tfidf"]
    else: 
        file_prefix = tfidf_dict["file_prefix"]
        tfidf_file_name = f"{file_prefix}_{end_date_name}_tfidf.parquet"
        save_path = os.path.join(
            tfidf_dict["fpath_tfidf"],
            tfidf_file_name
        )
    return save_path


def generate_tfidf_df(
    df_proc: pd.DataFrame,
    text_col_proc: str,
    data_key: str,
    save: bool = False
) -> pd.DataFrame:
    """
    Generate tfidf features for a single dataframe.
    """
    vectorizer = get_tfidf_vectorizer(
        text_col_proc=text_col_proc,
        save=True
    )
    _logger.info(
        "Transforming df_proc using fitted Tfidfvectorizer."
    )
    vectors = vectorizer.transform(df_proc[text_col_proc].values)
    feature_names = vectorizer.get_feature_names_out()
    df_tfidf = pd.DataFrame.sparse.from_spmatrix(
        vectors,
        columns=feature_names
    )
    df_tfidf = df_tfidf.sparse.to_dense()
    
    # Append target column if data_key is sentiment140.
    if data_key == "sentiment140":
        df_tfidf["target"] = (df_proc["target"] / 4).astype(int)
        cols = list(df_tfidf.columns)
        cols.remove("target")
        cols.insert(0, "target")
        df_tfidf = df_tfidf[cols]
    
    # Save results, if appropriate.
    if save:
        save_path = _get_tfidf_save_path(data_key=data_key)
        _logger.info(
            f"Saving tfidf features to {save_path}."
        )
        df_tfidf.to_parquet(save_path)
    return df_tfidf


def get_feature_generation_pipeline(
    tfidf_kwargs: dict,
    verbose: bool = False
):
    """
    Retreive an sklearn pipeline for feature generation.
    """
    tfidf = FunctionTransformer(
        generate_tfidf_df,
        kw_args=tfidf_kwargs,
        validate=False
    )
    return Pipeline(steps=[("tfidf", tfidf)], verbose=verbose)