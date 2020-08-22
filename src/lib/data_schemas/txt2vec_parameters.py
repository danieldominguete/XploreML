'''
===========================================================================================
Txt2Vec Model Building Parameters Class
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
    word2vec = 'word2vec'

class ApplicationType(str, Enum):
    '''

    '''
    word_embedding = 'word_embedding'

class Txt2VecParameters(BaseModel):

    # application
    application = 'language_modeling'

    # application type
    application_type:ApplicationType = 'word_embedding'

    # data source
    data_source: DataSource

    # train data file source
    data_train_file_path: Optional[str]

    # data file separator
    separator: str

    # percentual of data loading
    perc_load: float

    # mode of loading data
    mode_load: ModeLoad = ModeLoad.random

    # list of txt variables
    txt_inputs: list

    # list of numerical variables (not used)
    numerical_inputs: list = []

    # list of categorical variables (not used)
    categorical_inputs: list = []

    # output target classes (not used)
    output_target: list = []

    # type of modeling technique
    model_type: ModelType
