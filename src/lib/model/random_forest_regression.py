"""
===========================================================================================
Random Forest Regression Model Building Class
===========================================================================================
Script Reviewed by COGNAS
===========================================================================================
"""

import pandas as pd
import logging
from src.lib.model.xmodel import XModel
from sklearn.tree import DecisionTreeRegressor


class XRandomForesteRegression(XModel):
    def fit(self, data_input, data_target):

        # init model
        self.model = DecisionTreeRegressor(random_state=self.param.random_state)

        # fit model
        self.model.fit(data_input, data_target)

        # report results
        self.report_modeling()

        return self.model

    def eval_predict(self, data_input: pd) -> pd:

        # predict
        data_predict = self.model.predict(data_input)

        # convert ndarray to pandas dataframe
        data = pd.DataFrame(data=data_predict)

        return data

    def report_modeling(self) -> bool:

        logging.info("======================================================================")
        logging.info("Hyperparameters:")
        logging.info('random state: ' + str(self.param.random_state))
        logging.info('number of estimators: ' + str(self.param.n_estimators))

        return True
