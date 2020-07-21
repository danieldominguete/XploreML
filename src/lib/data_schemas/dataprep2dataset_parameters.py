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

class Dataprep2DatasetParameters(BaseModel):

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

    # list of numerical variables
    numerical_variables: list

    # list of categorical variables
    categorical_variables: list

    # list of txt variables
    txt_variables: list

    # output target (categorical ou number) - support only one column
    output_target: list

    # percentage of testing data
    test_subset: float

    # shuffle test data selection
    test_shuffle: bool
