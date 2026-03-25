import sys
from dataclasses import dataclass
import numpy as np
import os
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifact',"preprocessor.pkl")
    
class DataTransformation:
        def __init__(self):
            self.Data_Transformation_Config=DataTransformationConfig()
        def get_data_transformer_object(self):
            try:
                numerical_features=['writing score','reading score']
                cat_features=['gender',
                    'race/ethnicity',
                    'parental level of education',
                    'lunch',
                    'test preparation course'
                    ]
            except:
                pass
            
                
            
    

