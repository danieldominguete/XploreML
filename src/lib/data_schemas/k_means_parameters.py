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


class Init(str, Enum):
    '''
        "":
    '''
    kmeansplus = "k-means++"


class XKmeansParameters(BaseModel):
    n_clusters: int = 2,
    init: Init = "k-means++",
    random_state: int = 42
