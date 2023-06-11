"""
Utilities for working with s3.
"""


import boto3
import os
import pickle


from birdwatcher.config import PATHS


def s3_save_pickle(
    obj, 
    key: str, 
    file_name: str, 
    bucket: str = PATHS.s3_bucket.replace("s3://", "")
):
    """Save an object as a pickle file to s3."""
    s3_resource = boto3.resource("s3")
    key = key.replace(paths.s3_bucket + "/", "")
    key = os.path.join(key, file_name)
    save_obj_byte = pickle.dumps(obj)
    s3_resource.Object(bucket, key).put(Body=save_obj_byte)
    return


def s3_load_pickle(
    key: str, 
    file_name: str, 
    bucket: str = PATHS.s3_bucket.replace("s3://", "")
): 
    """Load an object from a pickle file in s3."""
    s3_resource = boto3.resource("s3")
    key = key.replace(paths.s3_bucket + "/", "")
    key = os.path.join(key, file_name)
    response = s3_resource.Bucket(bucket).Object(key).get()
    body_string = response["Body"].read()
    obj = pickle.loads(body_string)
    return obj