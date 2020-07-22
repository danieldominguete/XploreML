'''
===========================================================================================
K-nearest Neighbors Model Building Class
===========================================================================================
Script Reviewed by COGNAS
===========================================================================================
'''
import pandas as pd
import numpy as np
import logging
from src.lib.model.xmodel import XModel
from sklearn.neighbors import KNeighborsClassifier

class XKNearestNeighbors(XModel):

    def init(self) -> bool:

        # init model
        self._model = KNeighborsClassifier(n_neighbors=self._param.n_neighbors, metric=self._param.metric,
                                     p=self._param.metric_power)

        return True

    def fit(self, data_input, data_target):

        # init model
        self.init()

        # fit model
        self._model.fit(data_input, data_target)

        # report results
        self.save_results()

        return self.model

    def eval_predict(self, data_input:pd, int_to_cat_dict_target)->pd:

        raw_predict = self.model.predict(data_input)

        if self._application == "classification":
            data_predict = self.convert_classification_to_xout(data_predict=raw_predict, int_to_cat_dict_target=int_to_cat_dict_target)

        elif self._application == "regression":
            data_predict = self.convert_regression_to_xout(data_predict=raw_predict)

        return data_predict

    def save_results(self) -> bool:

        # hyperparameters (numbers and string)
        self._history['params'] = dict(self._param)

        # metrics (list numbers only)
        # self._history['metrics'] = {'teste': [100]}

        return True
