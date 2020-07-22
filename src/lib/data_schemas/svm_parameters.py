"""
===========================================================================================
SVM Regression Model Building Parameters Class
===========================================================================================
Script by COGNAS
===========================================================================================
"""
from pydantic import BaseModel
from enum import Enum
from typing import Optional

class Kernel(str, Enum):
    '''
        'linear regression':
        'sequential':
    '''
    rbr = 'rbf'

class XSVMParameters(BaseModel):

    # polynomial degree
    kernel: Kernel = 'rbf'
