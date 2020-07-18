'''
===========================================================================================
Dataset Parameters Class
===========================================================================================
Script by COGNAS
===========================================================================================
'''
from pydantic import BaseModel
from enum import Enum
from typing import Optional


class DatasetGoal(str, Enum):
    '''
        'supervised-learning':
    '''
    supervised_learning = 'supervised_learning'


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


class OutputType(str, Enum):
    '''
        "number":
        "binary_category":
        "multi_category_unilabel":
        "multi_category_multilabel":
    '''
    number = "number"
    binary_category = "binary_category"
    multi_category_unilabel = "multi_category_unilabel"
    multi_category_multilabel = "multi_category_multilabel"


class InputTxtFeaturesEmbedding(str, Enum):
    '''
        "":
        "word2int":
        "word2vec":
    '''
    none = ""
    word2int = "word2int"
    word2vec = "word2vec"


class ScalingNumberVariables(str, Enum):
    '''
        "":
        "min_max":
        "normal":
    '''
    none = ""
    min_max = "min_max"
    normal = "normal"


class EncodingCategoricalVariables(str, Enum):
    '''
        "":
        "one_hot":
        "int":
    '''
    none = ""
    one_hot = "one_hot"
    int = "int"


class Data2DataprepParameters(BaseModel):

    # data source
    data_source: DataSource

    # data file source
    data_file_path: str

    # data file separator
    separator: str

    # percentual of data loading
    perc_load: float

    # mode of loading data
    mode_load: ModeLoad = ModeLoad.random

    # list of numerical variables - support only numbers
    input_numerical_features: list

    # list of categorical variables - to categorical encode
    input_categorical_features: list

    # list of txt variables - to NLP embedding
    input_txt_features: list

    # output target (categorical ou number) - support only one column
    output_target: list

    # output target value type
    output_type: OutputType

    # scaling number variables:  None, "min-max" or "normal"
    scale_number_inputs: ScalingNumberVariables
    scale_number_outputs: ScalingNumberVariables

    # encoding categorical variables: "one-hot", "int"
    encode_categorical_inputs: EncodingCategoricalVariables
    encode_categorical_output: EncodingCategoricalVariables
