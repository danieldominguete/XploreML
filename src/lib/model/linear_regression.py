'''
===========================================================================================
Linear Regression Model Building Class
===========================================================================================
Script Reviewed by COGNAS
===========================================================================================
'''

import pandas as pd
import logging
from src.lib.model.xmodel import XModel
from sklearn.linear_model import LinearRegression

class XLinearRegression(XModel):


    def init(self) -> bool:

        # init model
        self._model = LinearRegression(fit_intercept=self._param.fit_intersection)

        return True

    def fit(self, data_input, data_target):

        # init model
        self.init()

        # fit model
        self._model.fit(data_input, data_target)

        # save results
        self.save_results()

        return True

    def eval_predict(self, data_input:pd)->pd:

        # model evaluation
        raw_predict = self._model.predict(data_input)

        if self._application == "classification":
            logging.error('Application not valid for Linear Regression.')

        elif self._application == "regression":
            data_predict = self.convert_regression_to_xout(data_predict=raw_predict)

        return data_predict
    
    def save_results(self) -> bool:

        # hyperparameters (numbers and string)
        self._history['params'] = dict(self._param)

        # metrics (list numbers only)
        # self._history['metrics'] = {'teste': [100]}

        return True

