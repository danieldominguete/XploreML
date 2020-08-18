"""
===========================================================================================
Recurrent Neural Model Building Parameters Class
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
    rnn_fccX = "RNN-LTSMx-FCCx"

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
    mean_squared_error = "mean_squared_error"

class ActivationFunctions(str, Enum):
    '''
        "":
    '''
    linear = "linear"
    sigmoid = "sigmoid"
    softmax = "softmax"
    relu = "relu"


class XNeuralRecurrentParameters(BaseModel):

    # framework implementation
    framework: Frameworks = "tensorflow"

    # recurrent topology id
    topology_id: Topology = "RNN-LTSMx-FCCx"

    # recurrent topology details
    # sublist for each input sequence always in order numerical - categorical - txt
    # example: [[A,B,C],[X,Y],[E,T]] => 3 sequences input. Seq1 has a stack of A-B-C nodes, seq2 has a stack of X-Y nodes, etc
    topology_details: dict = None

    # reduce learning rate activate
    reduce_lr: bool = True

    # early stopping activate
    early_stopping: bool = True

    # save checkpoints during training
    save_checkpoints: bool = True

    # loss
    loss_optim: LossesOptim = "mean_squared_error"

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