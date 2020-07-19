'''
===========================================================================================
Linear Regression Model Building Class
===========================================================================================
Script Reviewed by COGNAS
===========================================================================================
'''

import pandas as pd
from src.lib.model.model import XModel
from sklearn.linear_model import LinearRegression

class XLinearRegression(XModel):

    def fit(self, data_input, data_target):

        model = LinearRegression()
        self.model = model.fit(data_input, data_target)

        return model

    def eval_predict(self, data_input:pd)->pd:

        data_predict = self.model.predict(data_input)

        #convert ndarray to pandas dataframe
        data = pd.DataFrame(data=data_predict)

        return data

