import os
import sys
import dill


import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def modelevaluate(models,X_train,Y_train,X_test,Y_test):
    try:
        report={}
        for i in range(len(models)):
            model=list(models.values())[i]
            model.fit(X_train,Y_train)
            y_pred=model.predict(X_test)

            score=r2_score(Y_test,y_pred)

            report[list(models.keys())[i]]=score
        return report
    except Exception as e:
        raise CustomException(e,sys)

