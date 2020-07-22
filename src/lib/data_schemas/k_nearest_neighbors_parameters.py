"""
===========================================================================================
K-Nearest neighbors Model Building Parameters Class
===========================================================================================
Script by COGNAS
===========================================================================================
"""
from pydantic import BaseModel
from enum import Enum
from typing import Optional

class Metrics(str, Enum):
    '''
        "":
    '''
    minkowski = "minkowski"

class XKNearestNeighborsParameters(BaseModel):
    n_neighbors: int = 5,
    metric:Metrics = "minkowski",
    metric_power: int = 2
