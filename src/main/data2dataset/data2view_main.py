'''
===========================================================================================
Data2View : Data Visualization for Raw Data
===========================================================================================
'''
# =========================================================================================
# Importing the libraries
import logging
import os, sys, inspect
from dotenv import load_dotenv, find_dotenv
import argparse

# Include root folder to path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))
sys.path.insert(0,parentdir)

from src.lib.utils.util import Util
from src.lib.data_schemas.data2view_parameters import Data2ViewParameters
from src.lib.data_processing.data_processing import DataProcessing
from src.lib.data_schemas.environment_parameters import EnvironmentParameters
from src.lib.environment.environment import Environment
from src.lib.data_visualization.data_streamlit_visualization import DataVisualization

class ViewDataMain:

    def __init__(self, parameters_file):

        '''Constructor for this class'''
        self.parameters_file = parameters_file

    def run(self):

        # =========================================================================================
        # Environment variables
        load_dotenv(find_dotenv())
        PYTHON_WARNINGS = os.getenv("PYTHON_WARNINGS")

        # ===========================================================================================
        # Script Setup

        # Loading json file
        data_config = Util.load_parameters_from_file(path_file=self.parameters_file)

        # Validate parameters and load environment class
        env_param = EnvironmentParameters(**data_config.get("environment_parameters"))
        env = Environment(param=env_param)

        # Validade parameters and load raw data settings
        data_param = Data2ViewParameters(**data_config.get("view_data_parameters"))
        ds = DataProcessing(param=data_param)

        # ===========================================================================================
        # Setup environment
        env.init_script(script_name=os.path.basename(__file__), warnings_level=PYTHON_WARNINGS)

        # ===========================================================================================
        # Loading data
        logging.info("======================================================================")
        logging.info('Loading Raw Data:')
        data = ds.load_data()

        logging.info("======================================================================")
        logging.info('Update visualization of data:')
        dv = DataVisualization(title=env.param.app_name, data_param=data_param)
        dv.update_page(data=data)

        # ===========================================================================================
        # Script Performance
        env.close_script()
        # ===========================================================================================

# ===========================================================================================
# ===========================================================================================
# Main call from terminal
if __name__ == "__main__":
    '''
    Call from terminal command
    '''

    # getting script arguments
    parser = argparse.ArgumentParser(
        description='XploreML - Script Main for Dataset Vizualization'
    )
    parser.add_argument('-f', '--config_file_json',
                        help='Json config file for script execution', required=True)

    args = parser.parse_args()

    # Running script main
    try:
        processor = ViewDataMain(parameters_file=args.config_file_json)
        processor.run()
    except:
        logging.error('Ops ' + str(sys.exc_info()[0]) + ' occured!')
        raise
# ===========================================================================================
# ===========================================================================================