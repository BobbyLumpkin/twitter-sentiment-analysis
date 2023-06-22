"""
Utilities for working with s3.
"""


import awswrangler as wr
import boto3
import os
import pickle
import re
from typing import Optional

from birdwatcher.config import PATHS


def list_s3_elements(
    path: str,
    include_subfolders: bool = False,
    file_extensions: Optional[list[str]] = None
):
    """
    List all elements with a given s3 prefix.
    """
    if not path.endswith("/"):
        path += "/"
    
    objects = wr.s3.list_objects(path)
    directories = wr.s3.list_directories(path)
    elements = objects + directories
    
    if not include_subfolders:
        regexp = re.compile(path + ".+/.+$")
        elements = [
            ele for ele in elements
            if not regexp.search(ele)
        ]
    
    if file_extensions is not None:
        elements = [
            ele for ele in elements
            if ele.endswith(tuple(file_extensions))
        ]
    return elements


def list_s3_elements_with_pattern(
    path: str,
    pattern: str,
    include_subfolders: bool = False,
    file_extensions: Optional[list[str]] = None
) -> dict:
    """
    List all elements with a given s3 prefix and regex pattern.
    """
    elements = list_s3_elements(
        path=path,
        include_subfolders=include_subfolders,
        file_extensions=file_extensions
    )

    element_dict = {}
    for ele in elements:
        if re.search(pattern, ele):
            ele_stripped = ele.replace(path, "")
            match = re.search(pattern, ele_stripped).group()

            if match in element_dict.keys():
                match_list = element_dict[match]
                match_list.append(ele)
                element_dict.update({match: match_list})
            else:
                element_dict.update({match: [ele]})
    return element_dict


def s3_save_pickle(
    obj, 
    key: str, 
    file_name: str, 
    bucket: str = PATHS.s3_bucket.replace("s3://", "")
):
    """Save an object as a pickle file to s3."""
    s3_resource = boto3.resource("s3")
    key = key.replace(PATHS.s3_bucket + "/", "")
    key = os.path.join(key, file_name)
    save_obj_byte = pickle.dumps(obj)
    s3_resource.Object(bucket, key).put(Body=save_obj_byte)
    return


def s3_load_pickle(
    key: str, 
    file_name: str = None, 
    bucket: str = PATHS.s3_bucket.replace("s3://", "")
): 
    """Load an object from a pickle file in s3."""
    s3_resource = boto3.resource("s3")
    key = key.replace(PATHS.s3_bucket + "/", "")
    if file_name:
        key = os.path.join(key, file_name)
    response = s3_resource.Bucket(bucket).Object(key).get()
    body_string = response["Body"].read()
    obj = pickle.loads(body_string)
    return obj