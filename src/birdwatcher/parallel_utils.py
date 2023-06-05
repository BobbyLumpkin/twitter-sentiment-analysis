"""
Utilities for parallelization.
"""


import pandas as pd


def partition_indices(
    df: pd.DataFrame,
    num_intervals: int
) -> list:
    """
    Partition the indices of a dataframe.
    """
    num_rows = len(df.index)
    if num_intervals > num_rows:
        raise ValueError(
            f"'num_intervals' = {num_intervals}, "
            f"which is larger than the size of 'df' = {num_rows}."
        )
    if (num_rows % num_intervals) == 0:
        interval = int(num_rows / num_intervals)
    else:
        interval = int((num_rows // num_intervals) + 1)
    indices_list = []
    for i in range(0, num_rows - interval + 1, interval):
        indices = list(df.index[i : i + interval])
        indices_list.append(indices)
    if num_rows % interval != 0:
        indices = list(df.index[i + interval : ])
        indices_list.append(indices)
    return indices_list