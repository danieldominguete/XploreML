"""
===========================================================================================
Dataprep Parameters Class
===========================================================================================
Script by COGNAS
===========================================================================================
"""
from pydantic import BaseModel
from enum import Enum
from typing import Optional


class DataSource(str, Enum):
    """
        'localhost_datafile': local file
    """

    localhost_datafile = "localhost_datafile"


class ModeLoad(str, Enum):
    """
        'random':
        'sequential':
    """

    random = "random"
    sequential = "sequential"


class MissingFeaturesValues(str, Enum):
    """
        "delete": delete the sample
        “mean”, then replace missing values using the mean along each column. Can only be used with numeric data.
        “median”, then replace missing values using the median along each column. Can only be used with numeric data.
        “most_frequent”, then replace missing using the most frequent value along each column. Can be used with strings or numeric data.
        “constant”, then replace missing values with fill_value. Can be used with strings or numeric data
    """

    delete = "delete"
    mean = "mean"
    median = "median"
    most_frequent = "most_frequent"
    constant = "constant"


class OutliersMethods(str, Enum):
    """
        "k_std":
    """

    k_std = "k_std"


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

    # list of numerical variables
    numerical_variables: list

    # list of categorical variables
    categorical_variables: list

    # list of txt variables
    txt_variables: list

    # list of txt dict variables
    txt_dict_processing_variables: list

    # output target (categorical ou number) - support only one column
    output_target: list = []

    # missing features values : "delete"
    missing_number_inputs: MissingFeaturesValues = MissingFeaturesValues.delete
    missing_categorical_inputs: MissingFeaturesValues = MissingFeaturesValues.delete
    missing_txt_inputs: MissingFeaturesValues = MissingFeaturesValues.delete
    missing_outputs: MissingFeaturesValues = MissingFeaturesValues.delete

    # delete samples repeated (all columns)
    delete_repeated_samples: bool

    # ==================================
    # categorical specific preprocessing parameters
    # ==================================
    # list of list of variables values to maintain (following the categorical variables sequence)
    categorical_variables_include: list = []

    # list of list of variables values to exclude (following the categorical variables sequence)
    categorical_variables_exclude: list = []

    # ==================================
    # numerical specific preprocessing parameters
    # ==================================
    # remove outliers from numerical features
    remove_outliers: bool

    # outliers detection method
    outliers_method: OutliersMethods = OutliersMethods.k_std

    # Remove rows with column value is +/- k * std value
    k_numerical_outlier_factor: float
