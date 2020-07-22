"""
===========================================================================================
XploreML Model Class
===========================================================================================
Script Reviewed by COGNAS
===========================================================================================
"""

from abc import ABC, abstractmethod
import logging
import pandas as pd
import numpy as np


class XModel(ABC):
    def __init__(self, param: dict, application: str):
        self._param = param
        self._application = application
        self._model = None
        self._history = {}
        super().__init__()

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def fit(self, data_input: pd, data_target: pd):
        pass

    @abstractmethod
    def eval_predict(self, data_input: pd, application: str):
        pass

    @property
    def history(self) -> dict:
        return self._history

    @property
    def model(self):
        return self._model

    def convert_onehot_classification_to_xout(self, data_predict:pd, int_to_cat_dict_target:dict)-> pd:

        # convert coding output to label category (1 max)
        id_max = np.argmax(data_predict, axis=1)
        value_max = np.max(data_predict, axis=1)

        prediction = []
        prediction_reliability = []
        for sample in id_max:
            prediction.append(int_to_cat_dict_target[sample])

        for sample in value_max:
            prediction_reliability.append(sample)

        # convert ndarray to pandas dataframe
        data = pd.DataFrame(data= list(zip(prediction, prediction_reliability)), columns=["predict", "reliability"])

        return data

    def convert_regression_to_xout(self,data_predict:pd):

        prediction_reliability = np.ones((data_predict.shape[0],1), dtype=int)
        data = np.concatenate((data_predict, prediction_reliability), axis=1)

        # convert ndarray to pandas dataframe
        data = pd.DataFrame(data=data, columns=["predict", "reliability"])

        return data

    def print_hyperparameters(self) -> bool:
        if self._history is not None:
            data = self._history.get("params")
            for key, value in data.items():
                logging.info(str(key) + " : " + str(value))

        return True
