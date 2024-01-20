import pandas as pd
import numpy as np
import logging
from src.exception.exception import CustomException
import os
import sys
from sklearn.model_selection import train_test_split
from pathlib import Path
from dataclasses import dataclass 

@dataclass
class ModelEvaluationConfig:
    pass

class ModelEvaluation:

    def __init__(self):
        pass

    def initiate_model_evaluation(self):
        try:
            pass
        except Exception as e:
            logging.info("Error occurred")
            raise CustomException(e,sys) 
