"""
===========================================================================================
Feed Forward Neural Network Model Building Parameters Class
===========================================================================================
Script by COGNAS
===========================================================================================
"""
from pydantic import BaseModel
from enum import Enum
from typing import Optional, List

class Frameworks(str, Enum):
    '''
        "":
    '''
    tensorflow = "tensorflow"

class Topology(str, Enum):
    '''
        "":
    '''
    ffnn_fccX = "FFNN-FCCx"

class Optimizers(str, Enum):
    '''
        "":
    '''
    rmsprop = "rmsprop"
    adam = "adam"


class XNeuralFeedForwardParameters(BaseModel):

    # framework implementation
    framework: Frameworks = "tensorflow"

    # recurrent topology id
    topology_id: Topology = "FFNN-FCCx"

    # feed forward topology details
    # sublist for each input vector feature always in order: numerical - categorical - txt (only 1 vector feature)
    topology_details: dict = None

    # reduce learning rate activate
    reduce_lr: bool = True

    # early stopping activate
    early_stopping: bool = True

    # save checkpoints during training
    save_checkpoints: bool = True

    # optimizer
    optimizer: Optimizers = "rmsprop"

    # metrics
    metrics : list = ['mse']

    # batch size
    batch_size: int = 1

    # epochs
    epochs: int = 5

    # validation split
    validation_split = 0.2

    # verbose
    verbose:int = 0

    # shuffle
    shuffle:bool = True