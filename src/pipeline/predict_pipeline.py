import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.utilis import load_object


@dataclass
class PredictPipelineConfig:
    preprocessor_path: str = os.path.join("artifact", "preprocessor.pkl")
    model_path: str = os.path.join("artifact", "model.pkl")


class PredictPipeline:
    def __init__(self):
        self.config = PredictPipelineConfig()

    def predict(self, features: pd.DataFrame):
        try:
            preprocessor = load_object(self.config.preprocessor_path)
            model = load_object(self.config.model_path)

            data_scaled = preprocessor.transform(features)
            return model.predict(data_scaled)
        except Exception as e:
            raise CustomException(e, sys)


@dataclass
class CustomData:
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: float
    writing_score: float

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
