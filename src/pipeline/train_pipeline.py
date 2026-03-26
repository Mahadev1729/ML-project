import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException


def run_training_pipeline():
    try:
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_data, test_data
        )

        model_trainer = ModelTrainer()
        return model_trainer.initiate_model_trainer(train_arr, test_arr, preprocessor_path)
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    print(run_training_pipeline())
