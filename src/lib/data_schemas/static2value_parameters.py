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


class ModeType(str, Enum):
    '''
        'linear regression':
        'sequential':
    '''
    linear_regression = 'linear_regression'


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
        "":
        "one_hot":
        "int":
    '''
    none = ""
    one_hot = "one_hot"
    int = "int"


class Static2ValueParameters(BaseModel):
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

    # type of modeling technique
    model_type: ModeType

    # scaling numerical inputs
    scale_numerical_inputs: ScaleNumericalVariables = ''

    # encoding categorical variables: "one_hot", "int"
    encode_categorical_inputs: EncodingCategoricalVariables = ''
