"""
===========================================================================================
Dataprep2Dataset : Dataset for model building
===========================================================================================
"""
# =========================================================================================
# Importing the libraries
import logging
import os, sys, inspect
from dotenv import load_dotenv, find_dotenv
import argparse

# Include root folder to path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))
sys.path.insert(0, parentdir)

from src.lib.utils.util import Util
from src.lib.data_schemas.dataprep2dataset_parameters import Dataprep2DatasetParameters
from src.lib.data_schemas.environment_parameters import EnvironmentParameters
from src.lib.data_processing.data_processing import DataProcessing
from src.lib.environment.environment import Environment


class BuildDatasetMain:
    def __init__(self, parameters_file):

        """Constructor for this class"""
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

        # Validade parameters and load data processing class
        data_param = Dataprep2DatasetParameters(**data_config.get("dataset_parameters"))
        ds = DataProcessing(param=data_param)

        # ===========================================================================================
        # Setup environment
        env.init_script(script_name=os.path.basename(__file__), warnings_level=PYTHON_WARNINGS)

        # ===========================================================================================
        # Loading data
        logging.info("======================================================================")
        logging.info("Loading Raw Data:")
        data = ds.load_data()

        logging.info("======================================================================")
        logging.info("Split train and test data subsets:")
        data_train, data_test = ds.build_dataset(data=data)

        # ===========================================================================================
        # Analysis of data subsets

        logging.info("======================================================================")
        logging.info("Descritive Analysis - Training Data:")
        ds.descriptive_analysis(
            data=data_train,
            view_plots=env.param.view_plots,
            save_plots=env.param.save_plots,
            save_analysis=False,
            folder_path=env.run_folder,
            prefix=env.prefix_name,
        )

        logging.info("======================================================================")
        logging.info("Descritive Analysis - Test Data:")
        ds.descriptive_analysis(
            data=data_test,
            view_plots=env.param.view_plots,
            save_plots=env.param.save_plots,
            save_analysis=False,
            folder_path=env.run_folder,
            prefix=env.prefix_name,
        )

        # ===========================================================================================
        # Saving dataset
        logging.info("======================================================================")
        logging.info("Saving Datasets:")

        ds.save_datasets(
            data_train=data_train,
            data_test=data_test,
            folder_path=env.run_folder,
            prefix=env.prefix_name,
        )

        # ===========================================================================================
        # Register tracking info
        if env.param.tracking:
            env.publish_results(history=ds.history)

        # ===========================================================================================
        # Script Performance
        env.close_script()
        # ===========================================================================================


# ===========================================================================================
# ===========================================================================================
# Main call from terminal
if __name__ == "__main__":
    """
    Call from terminal command
    """

    # getting script arguments
    parser = argparse.ArgumentParser(
        description="XploreML - Script Main for Dataset Vizualization"
    )
    parser.add_argument(
        "-f", "--config_file_json", help="Json config file for script execution", required=True
    )

    args = parser.parse_args()

    # Running script main
    try:
        processor = BuildDatasetMain(parameters_file=args.config_file_json)
        processor.run()
    except:
        logging.error("Ops " + str(sys.exc_info()[0]) + " occured!")
        raise
# ===========================================================================================
# ===========================================================================================
