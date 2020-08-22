'''
===========================================================================================
K-means Model Building Class
===========================================================================================
Script Reviewed by COGNAS
===========================================================================================
'''
import pandas as pd
import numpy as np
import logging
from src.lib.model.xmodel import XModel
from sklearn.cluster import KMeans

class XKMeans(XModel):

    def init(self) -> bool:

        # init model
        self._model = KMeans(n_clusters = self._param.n_clusters, init = self._param.init, random_state = self._param.random_state)

        return True

    def fit(self, data_input):

        # init model
        self.init()

        # fit model
        self._model.fit(data_input)

        # report results
        self.save_results()

        return self.model

    def eval_predict(self, data_input:pd, int_to_cat_dict_target)->pd:

        raw_predict = self.model.predict(data_input)

        # if self._application == "classification":
        #     data_predict = self.convert_classification_to_xout(data_predict=raw_predict, int_to_cat_dict_target=int_to_cat_dict_target)
        #
        # elif self._application == "regression":
        #     data_predict = self.convert_regression_to_xout(data_predict=raw_predict)

        return True

    def save_results(self) -> bool:

        # hyperparameters (numbers and string)
        self._history['params'] = dict(self._param)

        # metrics (list numbers only)
        # self._history['metrics'] = {'teste': [100]}

        return True
