'''
===========================================================================================
Logistic Regression Model Building Class
===========================================================================================
Script Reviewed by COGNAS
===========================================================================================
'''

import pandas as pd
import logging
from src.lib.model.xmodel import XModel
from sklearn.linear_model import LogisticRegression

class XLogisticRegression(XModel):

    def fit(self, data_input, data_target):

        # init model
        model = LogisticRegression(random_state=self.param.random_state)
        
        # fit model
        if data_target.shape[1] == 1:
            self.model = model.fit(data_input, data_target)

            # report results
            self.report_modeling()
        else:
            logging.error('Logistic regression is not compatible with multi-category classification. Only binary.')

        return self.model

    def eval_regression_predict(self, data_input:pd)->pd:

        data_predict = self.model.predict(data_input)

        #convert ndarray to pandas dataframe
        data = pd.DataFrame(data=data_predict)

        return data
    
    def report_modeling(self) -> bool:

        logging.info("======================================================================")
        logging.info('Hyperparameters:')
        logging.info('random state : ' + str(self.param.random_state))
        #logging.info('fit_intersection : {a:.3f}'.format(a=self.param.fit_intersection))

        return True

