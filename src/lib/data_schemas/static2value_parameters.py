'''
===========================================================================================
Static2Value Model Building Parameters Class
===========================================================================================
Script by COGNAS
===========================================================================================
'''
from pydantic import BaseModel
from enum import Enum
from typing import Optional

class DataSource(str, Enum):
    '''
        'localhost_datafile': local file
    '''
    localhost_datafile = 'localhost_datafile'


class ModeLoad(str, Enum):
    '''
        'random':
        'sequential':
    '''
    random = 'random'
    sequential = 'sequential'


class ModelType(str, Enum):
    '''
        'linear regression':
    '''
    linear_regression = 'linear_regression'
    polynomial_regression = 'polynomial_regression'
    svm = 'svm'
    decision_tree = 'decision_tree'
    random_forest = 'random_forest'
    neural_dense = 'neural_dense'


class ScaleNumericalVariables(str, Enum):
    '''
        "":
        "min_max":
        "mean_std":
    '''
    none = ""
    min_max = "min_max"
    mean_std = "mean_std"


class EncodingCategoricalVariables(str, Enum):
    '''
        "one_hot":
        "int":
    '''
    one_hot = "one_hot"
    int = "int"


class Static2ValueParameters(BaseModel):

    # application
    application = 'regression'

    # data source
    data_source: DataSource

    # train data file source
    data_train_file_path: Optional[str]

    # test data file source
    data_test_file_path: Optional[str]

    # data file separator
    separator: str

    # percentual of data loading
    perc_load: float

    # mode of loading data
    mode_load: ModeLoad = ModeLoad.random

    # list of numerical variables
    numerical_inputs: list

    # list of categorical variables
    categorical_inputs: list

    # list of txt variables
    txt_inputs: list

    # output target number - support only one column
    output_target: list

    # scaling numerical inputs
    scale_numerical_inputs: ScaleNumericalVariables = ''

    # encoding categorical variables
    encode_categorical_inputs: EncodingCategoricalVariables = ''

    # scaling numerical output target
    scale_output_target: ScaleNumericalVariables = ''

    # type of modeling technique
    model_type: ModelType
