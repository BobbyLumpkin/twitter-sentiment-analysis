"""
Data prep of tweet data.
"""


from joblib import Parallel, delayed, cpu_count
import logging
from nltk.tokenize import word_tokenize
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from typing import Optional


from birdwatcher.config import DATA_PREP_CONFIG, PATHS
from birdwatcher.nlp_utils import (
    apply_word_lemmatizer,
    detect_wrapper,
    get_stop_words,
    join_text_list,
    remove_nonprintable_chars,
    remove_punctuation_from_list_regex,
    remove_twitter_lingo,
    remove_urls,
    remove_words_from_list,

)
from birdwatcher.parallel_utils import partition_indices


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


def _remove_and_tokenize(text: str) -> list:
    """
    Apply text cleaning to string.

    Args:
        text: The text to apply cleaning to.

    Returns:
        text: The original `text` with urls removed, twitter lingo
            removed, nonprintable characters removed, punctuation
            removed, stopwords removed and as a tokenized list.
    """
    words = get_stop_words()
    text = remove_words_from_list(
        remove_punctuation_from_list_regex(
            word_tokenize(
                remove_nonprintable_chars(
                    remove_twitter_lingo(
                        remove_urls(text.lower())
                    )
                )
            )
        ),
        words
    )
    return text


def _generate_raw_proc_path(
    config_dict: dict,
    df_type: str,
    end_date_name: str
):
    file_prefix = config_dict["file_prefix"]
    file_name = f"{file_prefix}_{end_date_name}_{df_type}.parquet"
    path = os.path.join(config_dict[f"fpath_{df_type}"], file_name)
    return path


def _load_df_if_not_provided(
    config_dict: Optional[dict] = None,
    df: Optional[pd.DataFrame] = None,
    df_type: Optional[str] = None,
    data_key: Optional[str] = None,
    end_date_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Load the appropriate pandas dataframe, if not provided.

    Args:
        config_dict: A dictionary containing `file_prefix` and
            `fpath_{df_type}` key-values, if `data_key` is not
            `sentiment140`. Else, only a `path` key-value is required.
        df: A pandas dataframe.
        df_type: One of `raw` or `proc`, indicating which type of
            dataframe to return.
        data_key: the data_key to fetch the dataframe for.
        end_date_name: The `end_date_name` to use for determining which
            dataframe to return.
    
    Returns:
        df: Either the provided dataframe or appropriate dataframe,
            loaded from cache.
    """
    # Return df, if provided.
    if df is not None:
        return df

    # Get df, if not provided.
    else:
        if data_key is None:
            raise TypeError(
                "Exactly one of 'df' or 'data_key' needs to not be "
                "'None'."
            )
        _logger.info(
            f"No dataframe provided. Loading {df_type} data associated "
            f"with data_key: '{data_key}'."
        )
        if data_key == "sentiment140":
            df = pd.read_parquet(config_dict[f"path_{df_type}"])
        else: 
            input_path = _generate_raw_proc_path(
                config_dict=config_dict,
                df_type=df_type,
                end_date_name=end_date_name
            )
            df = pd.read_parquet(input_path)
        return df


def _process_text_helper(
    df: pd.DataFrame,
    indices: list,
    text_col: str
) -> pd.DataFrame:
    """
    Process text column. Helper function for `process_text_df`.
    """
    # Define processed text column name and df subset
    text_col_proc = text_col + "_processed"
    df_tmp = df[df.index.isin(indices)].copy(deep=True)
    df_tmp["lang"] = df_tmp[text_col].apply(detect_wrapper).copy(deep=True)
    df_tmp = df_tmp[df_tmp.lang == "en"]

    # Remove urls, punctuation and stop words
    df_tmp[text_col_proc] = df_tmp[text_col].apply(lambda x : _remove_and_tokenize(x))

    # Lemmatize words
    df_tmp[text_col_proc] = df_tmp[text_col_proc].apply(apply_word_lemmatizer)
    df_tmp[text_col_proc] = np.asarray(df_tmp[text_col_proc])

    # Recombine tokenized words into a string
    df_tmp[text_col_proc] = df_tmp[text_col_proc].apply(lambda x : join_text_list(x))
    return df_tmp


def process_text_df(
    df: Optional[pd.DataFrame] = None,
    text_col: str = "text",
    data_key: Optional[str] = None, 
    n_jobs: int = -1, 
    verbose: int = 10, 
    end_date_name: str = run_info.end_date_name, 
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Clean the text column in a pandas dataframe.

    Args:
        text_col: The column name of the column with text to be cleaned.
            df: A pandas dataframe.
        data_key: A dictionary key of `dp_config.processing_info`.
        n_jobs: The number of cpus to use. (see `joblib.Parallel`
            documentation for more info.)
        verbose: The verbosity level. (see `joblib.Parallel`
            documentation for more info.)
        end_date_name: The date stamp, indicating which raw files to
            load, if `df` is `None`. (defaults to
            `run_info.end_date_name`.)
        save_path: Path to save the results to. (If not provided,
            results are not saved.)

    Returns:
        df: `df` with an additional column containing cleaned text from
            `df[text_col]`.
    """
    # Get df, if not provided.
    df = _load_df_if_not_provided(
        df=df,
        df_type="raw",
        data_key=data_key,
        end_date_name=run_info.end_date_name,
        config_dict=DATA_PREP_CONFIG.processing_info[data_key]
    )

    # Get indices list
    if n_jobs == -1: 
        num_intervals = cpu_count()
    else:
        num_intervals = n_jobs
    indices_list = partition_indices(df, num_intervals)
    
    # Run parallel processing
    df_list = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_process_text_helper)(
            df=df,
            indices=indices,
            text_col=text_col
        ) for indices in indices_list
    )
    df = pd.concat(objs=df_list, ignore_index=True)
    _logger.info(
        f"Raw text has been processed for data_key: '{data_key}'."
    )
    
    # Save, if applicable.
    if save_path:
        df.to_parquet(save_path)
        _logger.info(
            f"Processed data has been saved for data_key: {data_key}."
        )
    return df


def get_dataprep_pipeline(dataprep_kwargs: dict, verbose: bool = False):
    """
    Retreive an sklearn pipeline for dataprep.
    """
    dataprep = FunctionTransformer(
        process_text_df,
        kw_args=dataprep_kwargs,
        validate=False
    )
    return Pipeline(steps=[("dataprep", dataprep)], verbose=verbose)