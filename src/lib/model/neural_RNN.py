"""
===========================================================================================
Recurrent Neural Model Building Class
===========================================================================================
Script Reviewed by COGNAS
===========================================================================================
"""

import pandas as pd
import numpy as np
import logging
from src.lib.model.xmodel import XModel
from src.lib.model.tensorflow_model import XTensorFlowModel


class XNeuralRecurrent(XModel):
    def init(self) -> bool:

        # init model
        if self._param.framework == "tensorflow":
            self._model = XTensorFlowModel(param=self._param)
            self._model.set_topology_id("DNN-DENSE")
            self._model.set_run_folder_path(path=self._run_folder_path)
            self._model.set_prefix_name(prefix=self._prefix_name)
            self._model.set_tracking(value=self._tracking)
        else:
            logging.error("Framework not valid")

        return True

    def fit(self, data_input, data_target, input_var_dict, target_var_dict):

        # init model
        self.init()

        # convert pandas to numpy input and output
        X = self.convert_input_dataframe_to_tensors(dataframe=data_input, input_var_dict=input_var_dict, type="seq")
        Y = self.convert_output_dataframe_to_tensors(dataframe=data_target, output_var_dict=target_var_dict)

        # fit model
        self._model.fit(X, Y)

        # save results
        self.save_results()

        return True

    def convert_input_dataframe_to_tensors(self, dataframe:pd, input_var_dict:dict, type:str):

        if type == "static":

            number=False
            categorical=False
            txt=False

            # Reshape for list of one matrix [samples, features]
            input_feature_list = input_var_dict.get("number_inputs")
            if len(input_feature_list)>0:
                number = True
                for i in range(len(input_feature_list)):
                    if i == 0:
                        tensor_number = dataframe[input_feature_list[i]].to_numpy()
                    else:
                        tensor_aux = dataframe[input_feature_list[i]].to_numpy()
                        tensor_number = np.concatenate((tensor_number,tensor_aux),axis=1)

            input_feature_list = input_var_dict.get("categorical_inputs")
            if len(input_feature_list) > 0:
                categorical = True
                for i in range(len(input_feature_list)):
                    if i == 0:
                        tensor_categorical = dataframe[input_feature_list[i]].to_numpy()
                    else:
                        tensor_aux = dataframe[input_feature_list[i]].to_numpy()
                        tensor_categorical = np.concatenate((tensor_categorical, tensor_aux), axis=1)
                        
            if input_var_dict.get("txt_inputs") is not None:
                txt = True
                txt_in = input_var_dict.get("txt_inputs")
                for i in range(len(txt_in)):
                    if i == 0:
                        tensor_txt = dataframe[txt_in[i]].to_numpy(dtype=np.float32)
                    else:
                        tensor_txt_aux = dataframe[txt_in[i]].to_numpy(dtype=np.float32)
                        tensor_txt = np.concatenate((tensor_txt,tensor_txt_aux),axis=1)

            # concatenate inputs
            if number:
                tensor = tensor_number
                if categorical:
                    tensor = np.concatenate((tensor,tensor_categorical), axis=1)
                    if txt:
                        tensor = np.concatenate((tensor, tensor_txt), axis=1)
            else:
                if categorical:
                    tensor = tensor_categorical
                    if txt:
                        tensor = np.concatenate((tensor, tensor_txt), axis=1)
                else:
                    tensor = tensor_txt

            return [tensor]

        elif type == "seq":
            # Reshape for list of inputs with [samples, time steps, features] matrix
            # tensor = np.reshape(a=dataframe,newshape=(dataframe.shape[0],max_length_seq,1))

            tensor = []

            input_feature_list = input_var_dict.get("number_inputs")
            if len(input_feature_list) > 0:
                for i in range(len(input_feature_list)):
                    tensor_aux = dataframe[input_feature_list[i]].to_numpy()
                    tensor_aux = np.reshape(a=tensor_aux,newshape=(tensor_aux.shape[0],tensor_aux.shape[1],1))
                    tensor.append(tensor_aux)

            input_feature_list = input_var_dict.get("categorical_inputs")
            if len(input_feature_list) > 0:
                for i in range(len(input_feature_list)):
                    tensor_aux = dataframe[input_feature_list[i]].to_numpy()
                    tensor_aux = np.reshape(a=tensor_aux, newshape=(tensor_aux.shape[0],1, tensor_aux.shape[1]))
                    tensor.append(tensor_aux)

            input_feature_list = input_var_dict.get("txt_inputs")
            if len(input_feature_list) > 0:
                for i in range(len(input_feature_list)):
                    tensor_aux = dataframe[input_feature_list[i]].to_numpy()
                    tensor_aux = np.reshape(a=tensor_aux, newshape=(tensor_aux.shape[0], tensor_aux.shape[1],1))
                    tensor.append(tensor_aux)

            return tensor

    def convert_output_dataframe_to_tensors(self, dataframe: pd, output_var_dict: dict):

        # Reshape for TF DNN [samples, features]
        if output_var_dict.get("target_outputs") is not None:
            tensor = dataframe[output_var_dict.get("target_outputs")].to_numpy()

        return [tensor]

    def eval_predict(self, data_input: pd, int_to_cat_dict_target:dict = None) -> pd:

        # model evaluation
        raw_predict = self._model.predict(data_input)

        if self._application == "classification":
            data_predict = self.convert_classification_to_xout(data_predict=raw_predict,
                                                               int_to_cat_dict_target=int_to_cat_dict_target)

        elif self._application == "regression":
            data_predict = self.convert_regression_to_xout(data_predict=raw_predict.reshape(len(raw_predict), 1))

        return data_predict

    def save_results(self) -> bool:

        # hyperparameters (numbers and string)
        self._history["params"] = dict(self._param)

        # metrics (list numbers only)
        # self._history['metrics'] = {'teste': [100]}

        return True
