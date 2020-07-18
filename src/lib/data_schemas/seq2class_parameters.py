'''
===========================================================================================
Neural Parameters Class
===========================================================================================
Script by COGNAS
===========================================================================================
'''
from pydantic import BaseModel
from enum import Enum
from typing import Optional

class TopologyId(str, Enum):
    '''
        # name of topology
        # 'RNN-EMB-FIX-LSTM-DENSE' : Recurrent network with Embedding layer not trainable + 1 LSTM layer + 1 DENSE layer
        # 'RNN-EMB-TRAIN-LSTM-DENSE': Recurrent network with Embedding layer trainable + 1 LSTM layer + 1 DENSE layer
        # 'DNN-DENSE-DENSE-DENSE-DENSE': Dense network with 4 DENSE layers
        # 'RNN-LSTM-DENSE': Recurrent network with 1 LSTM layer + 1 DENSE layer
        # "RNN-CONV1D-MAXP1D-LSTM-DENSE": Recurrent network with 1D CONVOLUTION + 1D MAXPOOLING + 1 LSTM layer + 1 DENSE Layer
        # "RNN-LSTM-LSTM-DENSE": Recurrent network with 2 LSTM layers + 1 DENSE layer
        # "RNN-LSTM-LSTM-DENSE-Statefull": Recurrent network with 1 LSTM layer statefull + 1 DENSE layer
    '''
    RNN_EMB_FIX_LSTM_DENSE = "RNN-EMB-FIX-LSTM-DENSE"

class EmbeddingWordModelType(str, Enum):
    '''
        'word2vec_gensim': word2vec from gensim package
    '''
    word2vec_gensim = "word2vec_gensim"

class Seq2ClassParameters(BaseModel):

    # neural architeture
    topology_id : TopologyId

    # type of embedding model
    embedding_word_model_type : EmbeddingWordModelType

    # path to word embedding file
    embedding_word_model_file : str

    # nodes = number of neurons for each hidden layers
    nodes : list

    # func = number of hidden layers + 1 (output function)
    func : list

    # dropout tax for each layer
    dropout : list

    # size of batch data
    batch_size: int

    # max epochs
    epochs : int

    # percentage of validation subset
    validation_split : float

    # loss criteria for optimization algorithm
    loss : str

    # optimization algorithm
    optimizer : str

    # monitoring metrics
    metrics : list

    # verbose logging
    # 0: no logging
    # 1: print batch results and epoch results
    # 2: print only epoch results
    verbose : int

    # data selection for each epoch
    shuffle : bool

    # temporary files saved during training progress
    save_checkpoints : bool

    # early stopping criteria for regularization
    early_stopping : bool



