"""
===========================================================================================
Decision Tree Regression Model Building Parameters Class
===========================================================================================
Script by COGNAS
===========================================================================================
"""
from pydantic import BaseModel
from enum import Enum
from typing import Optional

class XRandomForestParameters(BaseModel):

    # random state
    random_state: int = 0

    # number of estimators
    n_estimators: int = 2
