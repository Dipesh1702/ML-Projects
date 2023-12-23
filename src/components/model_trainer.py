import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_traner(self,train_arr,test_arr):
        try:
            logging.info("Spliting traning and test info data")
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boost":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Negihbors Classification":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoostRegressor":CatBoostRegressor(verbose=False),
                "AdaBoostRegressor":AdaBoostRegressor()
             }
            Model_report:dict=evaluate_model(x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test,
                                             models=models)
            
            #to get best model score from dict
            best_model_score = max(sorted(Model_report.values()))

            # to get best model name from dict

            best_model_name = list(Model_report.keys())[
                list(Model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best Model found")
            logging.info(f"best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(x_test)

            r2_squqre = r2_score(y_test,predicted)

            return r2_squqre 
        except Exception as e:
            raise CustomException(e,sys)