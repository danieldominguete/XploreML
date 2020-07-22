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
        self.model = None
        super().__init__()

    @abstractmethod
    def fit(self, data_input, data_target):
        pass

    @abstractmethod
    def eval_regression_predict(self, data_input):
        pass

    @abstractmethod
    def eval_classification_predict(self, data_input):
        pass

    def test(self):
        print("ok!")
        return True