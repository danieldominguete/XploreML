'''
===========================================================================================
Data Parameters Class
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


class Data2ViewParameters(BaseModel):

    # data source
    data_source: DataSource

    # data file source
    data_file_path: str

    # data file separator
    separator: str

    # percentual of data loading
    perc_load: float

    # mode of loading data
    mode_load: ModeLoad = ModeLoad.sequential

    # list of numerical variables - support only numbers
    numerical_variables: list

    # list of categorical variables - to categorical encode
    categorical_variables: list

    # list of txt variables - to NLP embedding
    txt_variables: list