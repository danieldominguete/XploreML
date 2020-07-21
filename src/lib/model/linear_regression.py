'''
===========================================================================================
Linear Regression Model Building Class
===========================================================================================
Script Reviewed by COGNAS
===========================================================================================
'''

import pandas as pd
import logging
from src.lib.model.xmodel import XModel
from sklearn.linear_model import LinearRegression

class XLinearRegression(XModel):

    def fit(self, data_input, data_target):

        # init model
        model = LinearRegression(fit_intercept=self.param.fit_intersection)
        
        # fit model
        self.model = model.fit(data_input, data_target)

        # report results
        self.report_modeling()

        return model

    def eval_predict(self, data_input:pd)->pd:

        data_predict = self.model.predict(data_input)

        #convert ndarray to pandas dataframe
        data = pd.DataFrame(data=data_predict)

        return data
    
    def report_modeling(self):

        logging.info("======================================================================")
        logging.info('Hyperparameters:')
        logging.info('fit_intersection : ' + str(self.param.fit_intersection))
        #logging.info('fit_intersection : {a:.3f}'.format(a=self.param.fit_intersection))

