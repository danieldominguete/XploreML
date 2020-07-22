"""
===========================================================================================
Logistic Regression Model Building Parameters Class
===========================================================================================
Script by COGNAS
===========================================================================================
"""
from pydantic import BaseModel
from enum import Enum
from typing import Optional


class XLogisticRegressionParameters(BaseModel):

    random_state: int = 0
