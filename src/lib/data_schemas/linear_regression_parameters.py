"""
===========================================================================================
Linear Regression Model Building Parameters Class
===========================================================================================
Script by COGNAS
===========================================================================================
"""
from pydantic import BaseModel
from enum import Enum
from typing import Optional


class LinearRegressionParameters(BaseModel):

    # data source
    """Whether to calculate the intercept for this model. If set to False, no intercept will be used
    in calculations (i.e. data is expected to be centered). Independent term=0"""
    fit_intersection: bool
