'''
===========================================================================================
Logistic Regression Model Building Class
===========================================================================================
Script Reviewed by COGNAS
===========================================================================================
'''

import pandas as pd
import logging
from src.lib.model.xmodel import XModel
from sklearn.linear_model import LogisticRegression

class XLogisticRegression(XModel):

    def init(self) -> bool:

        # init model
        self._model = LogisticRegression(random_state=self._param.random_state)

        return True

    def fit(self, data_input, data_target):

        # init model
        self.init()
        
        # fit model
        if data_target.shape[1] == 1:
            self._model.fit(data_input, data_target)

            # save results
            self.save_results()
        else:
            logging.error('Logistic regression is not compatible with multi-category classification. Only binary.')

        return self._model

    def eval_predict(self, data_input: pd, int_to_cat_dict_target: dict = None) -> pd:

        # predict
        raw_predict = self.model.predict(data_input)

        if self._application == "classification":
            data_predict = self.convert_classification_to_xout(data_predict=raw_predict,
                                                               int_to_cat_dict_target=int_to_cat_dict_target)

        elif self._application == "regression":
            data_predict = self.convert_regression_to_xout(data_predict=raw_predict.reshape(len(raw_predict), 1))

        return data_predict

    def save_results(self) -> bool:

        # hyperparameters (numbers and string)
        self._history['params'] = dict(self._param)

        # metrics (list numbers only)
        # self._history['metrics'] = {'teste': [100]}

        return True