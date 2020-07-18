'''
===========================================================================================
XploreML Model Abstract Class
===========================================================================================
Script Reviewed by COGNAS
===========================================================================================
'''

from abc import ABC, abstractmethod

class XModel(ABC):

    def __init__(self, param:dict):
        self.param = param
        super().__init__()

    @abstractmethod
    def fit(self, data_input, data_target):
        pass