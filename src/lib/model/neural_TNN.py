"""
===========================================================================================
Neural Transformer Model Building Class
===========================================================================================
Script Reviewed by COGNAS
===========================================================================================
"""

import pandas as pd
import numpy as np
import logging
from src.lib.model.xmodel import XModel
from src.lib.model.tensorflow_model import XTensorFlowModel


class XNeuralTransformer(XModel):
    def init(self) -> bool:

        # init model
        if self._param.framework == "tensorflow":
            self._model = XTensorFlowModel(param=self._param)
            self._model.set_application(value=self._application)
            self._model.set_application_type(value=self._application_type.value)
            self._model.set_topology_id(topology=self._param.topology_id)
            self._model.set_topology_details(topology=self._param.topology_details[self._param.topology_id])
            self._model.set_run_folder_path(path=self._run_folder_path)
            self._model.set_prefix_name(prefix=self._prefix_name)
            self._model.set_tracking(value=self._tracking)

        else:
            logging.error("Framework not valid")

        return True

    def fit(self, data_input=None, data_target=None, input_var_dict=None, target_var_dict=None, target_cat_dict=None):

        # init model
        self.init()

        # convert pandas to numpy input and output
        X = self.convert_input_dataframe_to_tensors(dataframe=data_input, input_var_dict=input_var_dict, type="seq")
        Y = self.convert_output_dataframe_to_tensors(dataframe=data_target, output_var_dict=target_var_dict)

        # fit model
        self._model.fit(X=X, Y=Y, input_var_dict=input_var_dict, output_cat_dict=target_cat_dict)

        # save results
        self.save_results()

        return True

    def eval_predict(self, data_input: pd,input_var_dict:dict=None, int_to_cat_dict_target:dict = None) -> pd:

        # convert pandas to numpy input and output
        X = self.convert_input_dataframe_to_tensors(dataframe=data_input, input_var_dict=input_var_dict, type="seq")

        # model evaluation
        raw_predict = self._model.predict(X)

        if self._application == "classification":
            data_predict = self.convert_classification_to_xout(data_predict=raw_predict,
                                                               int_to_cat_dict_target=int_to_cat_dict_target[0])

        elif self._application == "regression":
            data_predict = self.convert_regression_to_xout(data_predict=raw_predict.reshape(len(raw_predict), 1))

        return data_predict

    def save_results(self) -> bool:

        # hyperparameters (numbers and string)
        self._history["params"] = dict(self._param)

        # training metrics (list numbers only)
        self._history['metrics'] = self.model._history.history

        return True
