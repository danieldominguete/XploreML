"""
===========================================================================================
Word2Vec Model Building Parameters Class
===========================================================================================
Script by COGNAS
===========================================================================================
"""
from pydantic import BaseModel
from enum import Enum
from typing import Optional

class Topology(str, Enum):
    '''

    '''
    cbow = 'cbow'
    skipgram = 'skipgram'

class XWord2VecParameters(BaseModel):

    # polynomial degree
    topology: Topology = 'cbow'

    # dimension of latent features
    encode_dim: int = 5

    # max distante between tokens for neighborhood
    window_max_distance:int = 2

    # epochs for training
    epochs:int = 5
