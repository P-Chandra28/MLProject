import os 
import sys
from dataclasses import dataclass

from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,modelevaluate

@dataclass
class ModelTrainingConfig:
    trained_model_filepath=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainingConfig()

    def Model_Training_Initiate(self,train_arr,test_arr):
        try:
            logging.info("Model training begins")
            X_train,Y_train,X_test,Y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
            "Linear Regression": LinearRegression(),
            "Lasso": Lasso(),
            "Ridge": Ridge(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "XGBRegressor": XGBRegressor(), 
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Regressor": AdaBoostRegressor()
            }

            modelreport: dict=modelevaluate(models=models,X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test)
            best_model_score=max((list(modelreport.values())))
            best_model_name=list(modelreport.keys())[
                list(modelreport.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            logging.info("best model obtained")
            save_object(
                file_path=self.model_trainer_config.trained_model_filepath,
                obj=best_model
            )
            Y_pred=best_model.predict(X_test)
            score=r2_score(Y_test,Y_pred)

            logging.info("model is trained")

            return score
        except Exception as e:
            raise CustomException(e,sys)