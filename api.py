"""
An API for sentiment classification model inference.
"""


from fastapi import FastAPI
import json
import pandas as pd
from pydantic import BaseModel, Field