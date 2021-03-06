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


class XPolynomialRegressionParameters(BaseModel):

    # polynomial degree
    degree: int
