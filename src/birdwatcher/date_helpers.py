"""
Purpose: Helper functions for dealing with dates and times in python.
Author(s): Bobby (Robert) Lumpkin
"""


from dateutil import relativedelta
from datetime import datetime
import pandas as pd


def string_to_date(date_str, format_in = '%Y-%m-%d'):
    """
    Convert a string to a datetime object.

    Parameters
    ----------
    date_str: A str to be converted to datetime.
    format_in: A string, indicating the format of 'date_str'.

    Returns
    ----------
    A conversion of 'date_str' to a datetime object.
    """
    if type(date_str) == datetime:
        return date_str
    elif type(date_str) == str:
        return datetime.strptime(date_str, format_in)
    else:
        raise TypeError("'date' must be of type str or datetime.")


def date_to_string(date_obj, format_out = '%Y-%m-%d'):
    """
   Convert a datetime object to a string.

   Parameters
   ----------
   date_obj: A datetime object to be converted to str.
   format_out: A string, indicating the format to convert 'date_obj' to.
    
    Returns
    ----------
    A conversion of 'date_obj' to a str object.
    """
    if type(date_obj) == str:
        return date_obj
    elif type(date_obj) == datetime:
        return datetime.strftime(date_obj, format_out)
    else:
        raise TypeError("'date' must be of type str or datetime.")
        

def convert_format(date, format_in, format_out):
    """
    Convert date string formats.

    Parameters
    ----------
    date: A date str or datetime object.
    format_in: A string, indicating the format of the input date (if
        'date' is of type str).
    format_out: A string, indicating the format to convert 'date' to.
    
    Returns
    ----------
    The original 'date' in the new format: 'format_out'.
    """
    if type(date) == str:
        date_obj = string_to_date(date, format_in)
    elif type(date) == datetime:
        date_obj = date
    else:
        raise TypeError("'date' must be of type str or datetime.")
        
    date_str = date_to_string(date_obj, format_out)
    return date_str


def first_date(
    date_str_list,
    format_in: str = "%Y-%m-%d",
    format_out: str = "Y-%m-%d"
):
    """
    Get first date in list of date strings.

    Parameters
    ----------
    date_str_list: A list or pandas series object of date strings.
    format_in: A string, indicating the format of the input date.
    format_out: A string, indicating the format to convert output to.
        Defaults is '%Y-%m-%d'.

    Returns
    ----------
    A date string representing the first (minimum) date in
        'date_str_list'.
    """
    if isinstance(date_str_list, pd.Series):
        date_str_list = date_str_list.tolist()
    if isinstance(date_str_list, list) == False:
        raise TypeError(
            "'date_str_list' must be a list or pandas series object."
        )
    if type(date_str_list[0]) == datetime:
        first_str = datetime.strftime(min(date_str_list), format_out)
    else:
        dt_list = [
            datetime.strptime(date, format_in)
            for date in date_str_list
        ]
        first_date = min(dt_list)
        first_str = first_dt.strftime(format_out)
    return first_str


def last_date(
    date_str_list,
    format_in: str = "%Y-%m-%d",
    format_out: str = "Y-%m-%d"
):
    """
    Get last date in list of date strings.

    Parameters
    ----------
    date_str_list: A list or pandas series object of date strings.
    format_in: A string, indicating the format of the input date.
    format_out: A string, indicating the format to convert output to.
        Defaults is '%Y-%m-%d'.

    Returns
    ----------
    A date string representing the last (maximum) date in
        'date_str_list'.
    """
    if isinstance(date_str_list, pd.Series):
        date_str_list = date_str_list.tolist()
    if isinstance(date_str_list, list) == False:
        raise TypeError(
            "'date_str_list' must be a list or pandas series object."
        )
    if type(date_str_list[0]) == datetime:
        last_str = datetime.strftime(max(date_str_list), format_out)
    else:
        dt_list = [
            datetime.strptime(date, format_in)
            for date in date_str_list
        ]
        last_date = max(dt_list)
        last_str = last_dt.strftime(format_out)
    return last_str


def add_days(
    date: str,
    days: int,
    format_in: str="%Y-%m-%d",
    format_out: str="%Y-%m-%d",
    return_type="str"
):
    """
    Add specified number of days to a date.
    """
    if type(date) == str:
        datetime_object = datetime.strptime(date, format_in)
    elif type(date) == datetime:
        datetime_object = date
    else:
        raise TypeError("'date' must be of type str or datetime.")
    
    d_days = relativedelta.relativedelta(days=days)
    delta_date_dt = datetime_object + d_days

    if return_type == "str":
        return delta_date_dt.strftime(format_out)
    else:
        return delta_date_dt 



def compute_shifted_month_end(
    date,
    delta,
    unit: str = "m",
    format_in: str = "%Y-%m-%d",
    format_out: str = "Y-%m-%d",
    return_type=str
):
    """
    Get a relative month-end date.

    Parameters
    ----------
    date: Either a date string or datetime object.
    delta: A float, indicating the relative delta to add to 'date'.
    unit: A string, indicating the unit for delta. One of 'd', 'm', 'Y'.
    format_in: A string, indicating the format of the input date.
    format_out: A string, indicating the format to convert output to.
        Defaults is '%Y-%m-%d'.
    return_type: A type (either str or datetime.datetime) indicating
        what type of object to return.
    
    Returns
    ----------
    The month-end date of the month 'delta' 'unit's relative to 'date'.
    """
    if type(date) == str:
        datetime_object = datetime.strptime(date, format_in)
    elif type(date) == datetime:
        datetime_object = date
    else:
        raise TypeError("'date' must be of type str or datetime.")
    if unit not in ["d", "m", "Y"]:
        raise ValueError("'unit' must be one of 'd', 'm', 'Y'.")
    
    if unit == "Y":
        d_year = relativedelta.relativedelta(years=delta)
        d_day = relativedelta.relativedelta(day=31)
        delta_date_dt = datetime_object + d_year + d_day
        delta_date_str = delta_date_dt.strftime(format_out)
    elif unit == "m":
        d_month = relativedelta.relativedelta(months=delta)
        d_day = relativedelta.relativedelta(day=31)
        delta_date_dt = datetime_object + d_month + d_day
        delta_date_str = delta_date_dt.strftime(format_out)
    else:
        d_day = relativedelta.relativedelta(days=delta)
        d_day1 = relativedelta.relativedelta(day=31)
        delta_date_dt = datetime_object + d_day + d_day1
        delta_date_str = delta_date_dt.strftime(format_out)
    
    if return_type == str:
        return delta_date_str
    else:
        return delta_date_dt
