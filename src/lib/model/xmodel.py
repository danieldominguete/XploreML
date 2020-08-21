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
from src.lib.environment.environment import Environment

class XModel(ABC):
    def __init__(self, param: dict, application: str, application_type: str = None, env: Environment = None):
        self._param = param
        self._application = application
        self._application_type = application_type
        self._run_folder_path = env.run_folder
        self._prefix_name = env.prefix_name
        self._tracking = env.tracking
        self._model = None
        self._history = {}

        super().__init__()

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def fit(self, data_input: pd, data_target: pd, input_var_dict:dict, target_var_dict:dict):
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

    def convert_classification_to_xout(self, data_predict:pd, int_to_cat_dict_target:dict)-> pd:

        if self._application_type == "multi_category_unilabel":
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

        if self._application_type == "binary_category":

            # convert coding output to label category
            prediction = []
            prediction_reliability = []
            value_predict = np.max(data_predict, axis=1)

            # getting round between 0 and 1 for label and reliability
            for value in value_predict:
                prediction_reliability.append(value)
                class_bin = int(np.round(value,0))
                prediction.append(int_to_cat_dict_target.get(class_bin))


            # convert ndarray to pandas dataframe
            data = pd.DataFrame(data=list(zip(prediction, prediction_reliability)), columns=["predict", "reliability"])

            return data

        logging.error('Application type not valid for convertion.')
        return None

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

    def convert_input_dataframe_to_tensors(self, dataframe: pd, input_var_dict: dict, type: str):

        if type == "static":

            number = False
            categorical = False
            txt = False

            # Reshape for list of one matrix [samples, features]
            input_feature_list = input_var_dict.get("number_inputs")
            if len(input_feature_list) > 0:
                number = True
                for i in range(len(input_feature_list)):
                    if i == 0:
                        tensor_number = dataframe[input_feature_list[i]].to_numpy()
                    else:
                        tensor_aux = dataframe[input_feature_list[i]].to_numpy()
                        tensor_number = np.concatenate((tensor_number, tensor_aux), axis=1)

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
                        tensor_txt = np.concatenate((tensor_txt, tensor_txt_aux), axis=1)

            # concatenate inputs
            if number:
                tensor = tensor_number
                if categorical:
                    tensor = np.concatenate((tensor, tensor_categorical), axis=1)
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
                    tensor_aux = np.reshape(a=tensor_aux, newshape=(tensor_aux.shape[0], tensor_aux.shape[1], 1))
                    tensor.append(tensor_aux)

            input_feature_list = input_var_dict.get("categorical_inputs")
            if len(input_feature_list) > 0:
                for i in range(len(input_feature_list)):
                    tensor_aux = dataframe[input_feature_list[i]].to_numpy()
                    tensor_aux = np.reshape(a=tensor_aux, newshape=(tensor_aux.shape[0], 1, tensor_aux.shape[1]))
                    tensor.append(tensor_aux)

            input_feature_list = input_var_dict.get("txt_inputs")
            if len(input_feature_list) > 0:
                for i in range(len(input_feature_list)):
                    tensor_aux = dataframe[input_feature_list[i]].to_numpy()
                    # tensor_aux = np.reshape(a=tensor_aux, newshape=(tensor_aux.shape[0], tensor_aux.shape[1],1))
                    tensor.append(tensor_aux)

            return tensor

    def convert_output_dataframe_to_tensors(self, dataframe: pd, output_var_dict: dict):

        # Reshape for TF DNN [samples, features]
        if output_var_dict.get("target_outputs") is not None:
            tensor = dataframe[output_var_dict.get("target_outputs")].to_numpy()

        return [tensor]