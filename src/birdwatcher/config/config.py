"""
Purpose: Config class definition and instantiations.
"""


import boto3
import os
from pathlib import Path
import yaml


class PathConcatenator(yaml.YAMLObject): 
    yaml_loader = yaml.SafeLoader
    yaml_tag = "!osjoin"
    
    @classmethod
    def from_yaml(cls, loader, node): 
        seq = loader.construct_sequence(node)
        return os.path.join(*seq)


class StringConcatenator(yaml.YAMLObject):
    yaml_loader = yaml.SafeLoader
    yaml_tag = "!strjoin"

    @classmethod
    def from_yaml(cls, loader, node):
        seq = loader.construct_sequence(node)
        return ''.join(seq)


class PathsTranslator(yaml.YAMLObject):
    yaml_loader = yaml.SafeLoader
    yaml_tag = "!paths"

    @classmethod
    def from_yaml(cls, loader, node):
        key = loader.construct_scalar(node)
        path = getattr(PATHS, key)
        return path
            

class Config:
    def __init__(self, rel_path, bucket=None):
        self.base_path = Path(os.path.realpath(__file__)).parents[1].absolute()
        self.config_yaml = self.base_path / rel_path
        self.dc_run_date_info = (
            self.base_path / "config/collection_run_info.pkl"
        )
        self.tfidf_vectorizer_path = self.base_path / "ml/tfidf_vectorizer.pkl"
        self.training_pipeline_path = (
            self.base_path / "ml/training_pipeline.pkl"
        )
        self.pca_path = self.base_path / "ml/trained_pca.pkl"
        self.model_path = self.base_path / "ml/trained_model.pkl"

        if bucket:
            self.bucket = Path(bucket)
            self.config_yaml = self.bucket / rel_path
        self.reset_config(rel_path, bucket)
        return
    
    def _process(self, **params): 
        for key, value in params.items(): 
            setattr(self, key, value)
    
    def reset_config(self, rel_path, bucket):
        if bucket:
            s3_client = boto3.client("s3")
            response = s3_client.get_object(
                Bucket=bucket,
                Key=rel_path
            )
            config = yaml.safe_load(response["Body"])
            self._process(**config)
            return
        with open(self.config_yaml, "rb") as f: 
            config = yaml.safe_load(f.read())
            self._process(**config)
            return


PATHS = Config("config/paths.yaml")
SECRETS = Config(
    rel_path="assets/secrets.yaml",
    bucket="twitter-sentiment-analysis"
)
DATA_COLLECTION_CONFIG = Config("config/data_collection_config.yaml")
DATA_PREP_CONFIG = Config("config/dataprep_config.yaml")
FEATURE_GENERATION_CONFIG = Config("config/feature_generation_config.yaml")
SUMMARIZE_CONFIG = Config("config/summarize_config.yaml")
TRAINING_CONFIG = Config("config/training_config.yaml")