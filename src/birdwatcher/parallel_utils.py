"""
Utilities for parallelization.
"""


from dataclasses import dataclass, field
import pandas as pd
import warnings


@dataclass
class IntervalsExceedsDataFrame(Warning):
    
    """
    Warning for when number of intervals exceeds the length of the df.
    """

    message: str = field(default=(
        "The number of intervals exceeds the length of the "
        "DataFrame. Reverting to the DataFrame length for the "
        "number of intervals."
    ))


def partition_indices(
    df: pd.DataFrame,
    num_intervals: int
) -> list:
    """
    Partition the indices of a dataframe.
    """
    num_rows = len(df.index)
    if num_intervals > num_rows:
        warnings.warn(
            f"'num_intervals' = {num_intervals}, "
            f"which is larger than the size of 'df' = {num_rows}.",
            IntervalsExceedsDataFrame
        )
        num_intervals = num_rows
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