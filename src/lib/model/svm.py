"""
===========================================================================================
SVM Regression Model Building Class
===========================================================================================
Script Reviewed by COGNAS
===========================================================================================
"""

import pandas as pd
import logging
from src.lib.model.xmodel import XModel
from sklearn.svm import SVR, SVC


class XSVM(XModel):

    def init(self) -> bool:

        ## init model
        if self._application == "regression":
            self._model = SVR(kernel=self._param.kernel)
        elif self._application == "classification":
            self._model = SVC(kernel=self._param.kernel)
        else:
            logging.error('Application type for SVM is not valid')

        return True

    def fit(self, data_input, data_target):

        # init model
        self.init()

        # fit model
        self._model.fit(data_input, data_target)

        # save results
        self.save_results()

        return self._model

    def eval_predict(self, data_input: pd, int_to_cat_dict_target:dict = None) -> pd:

        # predict
        raw_predict = self.model.predict(data_input)

        if self._application == "classification":
            data_predict = self.convert_onehot_classification_to_xout(data_predict=raw_predict, int_to_cat_dict_target=int_to_cat_dict_target)

        elif self._application == "regression":
            data_predict = self.convert_regression_to_xout(data_predict=raw_predict.reshape(len(raw_predict),1))

        return data_predict

    def save_results(self) -> bool:

        # hyperparameters (numbers and string)
        self._history['params'] = dict(self._param)

        # metrics (list numbers only)
        # self._history['metrics'] = {'teste': [100]}

        return True

