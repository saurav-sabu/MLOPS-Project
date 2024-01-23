import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
import pickle
from src.utils.utils import load_object
from urllib.parse import urlparse
from src.logger.logger import logging
from src.exception.exception import CustomException

from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error


class ModelEvaluation:

    def __init__(self):
        logging.info("Evaluation started")

    def eval_metrics(self,actual,pred):
        rmse = np.sqrt(mean_squared_error(actual,pred))
        mae = mean_absolute_error(actual,pred)
        r2 = r2_score(actual,pred)
        logging.info("Metrics captured")
        return rmse,mae,r2

    def initiate_model_evaluation(self,train_arr,test_arr):
        try:
            X_test,y_test = (test_arr[:,:-1],test_arr[:,-1])
            model_path = os.path.join("artifacts","model.pkl")

            model = load_object(model_path)

            # mlflow.set_registry_uri("")

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            print(tracking_url_type_store)

            with mlflow.start_run():
                prediction = model.predict(X_test)

                (rmse,mae,r2) = self.eval_metrics(y_test, prediction)

                mlflow.log_metric("rmse",rmse)
                mlflow.log_metric("mae",mae)
                mlflow.log_metric("r2",r2)

                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model, "model")


        
        except Exception as e:
            raise CustomException(e, sys)

