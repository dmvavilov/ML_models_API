#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import os

from sklearn.linear_model import LinearRegression, LogisticRegression
from lightgbm import LGBMRegressor, LGBMClassifier


# In[10]:


class ML_models:
    def __init__(self):
        
        self.available_models = {'Regression':  {'LinReg': LinearRegression, 
                                                 'LGBM': LGBMRegressor},
                                 'Classification': {'LogReg': LogisticRegression, 
                                                    'LGBM': LGBMClassifier}}
        
        self.fitted_models = {} # будем хранить имя модели и зафиченную модельку для каждого пользователя

    def get_available_tasks(self):
        return list(self.available_models.keys())

    def get_available_models(self, task:str):
        if task == None:
            return 'You should pass task first'
        if task not in self.get_available_tasks():
            return f'Task type {task} is not available'
        return list(self.available_models[task].keys())

    def fit_model(self, task, model, model_name, X, y, upd=False, params={}):
        X = pd.DataFrame.from_dict(X)
        y = pd.DataFrame.from_dict(y)
        if (model_name in self.fitted_models.keys()) and upd==False:
            return 'Such model name already exists. You can delete/update it or choose another name'

        model_ml = self.available_models[task][model](**params)
        model_ml.fit(X.values, y.values.flatten())
        self.fitted_models[model_name] = model_ml

        return f'Model successfully fitted. Score: {model_ml.score(X.values, y.values.flatten())}'

    def delete_model(self, model_name):
        if model_name not in self.fitted_models.keys():
            return 'No model with such name was fitted'
        del self.fitted_models[model_name]
        return f'Model {model_name} was successfully deleted'

    def get_preds(self, model_name, X_test):
        X_test = pd.DataFrame.from_dict(X_test)
        index = X_test.index
        if model_name not in self.fitted_models.keys():
            return 'No model with such name was fitted'
        return dict(zip(index.values, self.fitted_models[model_name].predict(X_test)))
        

