'''
===========================================================================================
Neural Parameters Class
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

    '''
    neural_recurrent = 'neural_recurrent'


class EncodingNumericalVariables(str, Enum):
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

class EncodingTxtVariables(str, Enum):
    '''

    '''
    int = "word2int"

class ClassificationType(str, Enum):
    '''
        "":
    '''
    binary_category = "binary_category"
    multi_category_unilabel = "multi_category_unilabel"
    multi_category_multilabel = "multi_category_multilabel"  # not implemented

class Seq2ClassParameters(BaseModel):

    # application
    application = 'classification'

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

    # list of numerical features. Each feature is a sublist of time steps values.
    numerical_inputs: list

    # list of categorical variables. Each feature is a sublist of one value.
    categorical_inputs: list

    # list of txt variables. Each feature is a sublist of concatenated variables.
    txt_inputs: list

    # output target classes - support only one column
    output_target: list

    # classification type
    classification_type: ClassificationType = None

    # scaling numerical inputs
    scale_numerical_inputs: EncodingNumericalVariables = ''

    # encoding categorical variables
    encode_categorical_inputs: EncodingCategoricalVariables = ''

    # encoding txt variables
    encode_txt_inputs: EncodingTxtVariables = "int"

    # max token
    txt_inputs_max_length: int = 10

    # type of modeling technique
    model_type: ModelType

    # output encoding
    encode_output: str =""