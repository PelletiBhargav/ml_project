import os
import sys
from dataclasses import dataclass

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("spliting training and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                        "Decision Tree": DecisionTreeRegressor(),
                        "Random Forest": RandomForestRegressor(), 
                        "Gradient Boosting": GradientBoostingRegressor(),
                        "Linear Regression": LinearRegression(),
                        
                          # ✅ fixed
                          # ✅ added
                        "XGBRegressor": XGBRegressor(),
                        "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                        "AdaBoost Regressor": AdaBoostRegressor(),
                        "Lasso": Lasso(),
                        "Ridge": Ridge(),
                        "K-Neighbors Regressor": KNeighborsRegressor()
                    }

            
            params = {
                    "Decision Tree": {
                        'criterion': ['squared_error', 'friedman_mse']
                    },
                    "Random Forest": {
                        'n_estimators': [64, 128, 256]
                    },
                    "Gradient Boosting": {
                        'learning_rate': [0.01, 0.05, 0.1],
                        'n_estimators': [64, 128]
                    },
                    "Linear Regression": {},
                    "XGBRegressor": {
                        'learning_rate': [0.01, 0.1],
                        'n_estimators': [64, 128]
                    },
                    "CatBoosting Regressor": {
                        'depth': [6, 8],
                        'learning_rate': [0.01, 0.1],
                        'iterations': [50, 100]
                    },
                    "AdaBoost Regressor": {
                        'learning_rate': [0.01, 0.1],
                        'n_estimators': [64, 128]
                    },
                    "Lasso": {},
                    "Ridge": {},
                    "K-Neighbors Regressor": {}
                }

            
            
            model_report, best_model = evaluate_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                params=params
            )

            best_model_score = max(model_report.values())

            if best_model_score < 0.60:
                raise CustomException("No best model found")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

            
            
        except Exception as e:
            raise CustomException(e,sys)