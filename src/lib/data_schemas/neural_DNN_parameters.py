"""
===========================================================================================
Dense Neural Model Building Parameters Class
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

class Optimizers(str, Enum):
    '''
        "":
    '''
    rmsprop = "rmsprop"
    adam = "adam"

class LossesOptim(str, Enum):
    '''
        "":
    '''
    binary_crossentropy = "binary_crossentropy"
    categorical_crossentropy = "categorical_crossentropy"
    sparse_categorical_crossentropy = "sparse_categorical_crossentropy"
    mean_square_error = "mean_square_error"

class ActivationFunctions(str, Enum):
    '''
        "":
    '''
    linear = "linear"
    sigmoid = "sigmoid"
    softmax = "softmax"
    relu = "relu"


class XNeuralDenseParameters(BaseModel):

    # framework implementation
    framework: Frameworks = "tensorflow"

    # hidden nodes
    hidden_nodes: list = [1]

    # nodes functions
    hidden_func_nodes: List[ActivationFunctions] = ["relu"]

    # dropout value for each layer
    hidden_dropout: list = [0.1]

    # output layer nodes function
    output_func_nodes: ActivationFunctions = "linear"

    # reduce learning rate activate
    reduce_lr: bool = True

    # early stopping activate
    early_stopping: bool = True

    # save checkpoints during training
    save_checkpoints: bool = True

    # loss
    loss_optim: LossesOptim = "mean_square_error"

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