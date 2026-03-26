import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utilis import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def _get_param_distributions(self, model_name: str):
        if model_name == "Random Forest":
            return {
                "n_estimators": [100, 200, 400, 800],
                "max_depth": [None, 5, 10, 20, 40],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
            }
        if model_name == "Gradient Boosting":
            return {
                "n_estimators": [100, 200, 400],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [2, 3, 4, 5],
                "subsample": [0.6, 0.8, 1.0],
            }
        if model_name == "Decision Tree":
            return {
                "max_depth": [None, 5, 10, 20, 40],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
            }
        if model_name == "K-Neighbors Regressotr":
            return {
                "n_neighbors": [3, 5, 7, 11, 15, 21],
                "weights": ["uniform", "distance"],
                "p": [1, 2],
                "leaf_size": [10, 20, 30, 40, 50],
            }
        if model_name == "XGBRegressor":
            return {
                "n_estimators": [200, 400, 800],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 4, 5, 7, 10],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "reg_alpha": [0.0, 0.1, 1.0],
                "reg_lambda": [1.0, 5.0, 10.0],
            }
        if model_name == "CatBoosting Regressor":
            return {
                "depth": [4, 6, 8, 10],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "iterations": [300, 600, 900],
                "l2_leaf_reg": [1, 3, 5, 7, 9],
            }
        if model_name == "AdaBoost Regressor":
            return {
                "n_estimators": [50, 100, 200, 400],
                "learning_rate": [0.01, 0.05, 0.1, 0.2, 1.0],
                "loss": ["linear", "square", "exponential"],
            }
        return None

    def _tune_model(self, model, model_name: str, X_train, y_train):
        param_distributions = self._get_param_distributions(model_name)
        if not param_distributions:
            model.fit(X_train, y_train)
            return model, None

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=25,
            scoring="r2",
            cv=3,
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        return search.best_estimator_, search.best_params_
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbors Regressotr":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(verbose=False),
                "AdaBoost Regressor":AdaBoostRegressor(),
            }
            
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            
            best_model_score=max(sorted(model_report.values()))
            
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found", sys)

            best_model, best_params = self._tune_model(best_model, best_model_name, X_train, y_train)

            logging.info(f"Best found model on both training and testing dataset")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            model_r2_score=r2_score(y_test,predicted)
            
            return model_r2_score
        except Exception as e:
            raise CustomException(e,sys)
            
        
