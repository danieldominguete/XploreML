"""
===========================================================================================
Static2Cluster : Model building from static data to clustering
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
from src.lib.data_schemas.static2cluster_parameters import Static2ClusterParameters
from src.lib.data_schemas.environment_parameters import EnvironmentParameters
from src.lib.data_processing.data_processing import DataProcessing
from src.lib.environment.environment import Environment
from src.lib.data_schemas.k_means_parameters import XKmeansParameters
from src.lib.model.k_means import XKMeans
from src.lib.model.model_evaluation import ClassificationModelEvaluation


class BuildStatic2ValueMain:
    def __init__(self, parameters_file):

        """Constructor for this class"""
        self.parameters_file = parameters_file

    def model_selection(self, data_config, data_param):

        # Validade and load model hyperparameters
        if data_param.model_type == "k_means":
            model_param = XKmeansParameters(
                **data_config.get("k_means_parameters")
            )
            model = XKMeans(param=model_param, application=data_param.application)

        else:
            logging.error('Model type not valid.')

        return model

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
        data_param = Static2ClusterParameters(**data_config.get("static2cluster_parameters"))
        ds = DataProcessing(param=data_param)

        # ===========================================================================================
        # Setup environment
        env.init_script(script_name=os.path.basename(__file__), warnings_level=PYTHON_WARNINGS)

        # ===========================================================================================
        # Loading data
        logging.info("======================================================================")
        logging.info("Loading Training and Test Data:")
        data_train_input, _ = ds.load_train_data()
        data_test_input, _ = ds.load_test_data()

        logging.info("======================================================================")
        logging.info("Preprocessing Training Data:")
        (
            data_train_input,
            _,
            data_test_input,
            _,
            variables_input,
            _,
            _,
            _,
        ) = ds.prepare_train_test_data(
            data_train_input=data_train_input,
            data_test_input=data_test_input
        )

        logging.info("======================================================================")
        logging.info("Building Model:")

        model = self.model_selection(data_config=data_config, data_param=data_param)

        model.fit(
            data_input=data_train_input[variables_input]
        )

        logging.info("======================================================================")
        logging.info("Building predictions:")

        # TODO

        logging.info("======================================================================")
        logging.info("Training Results")

        # TODO

        logging.info("======================================================================")
        logging.info("Test Results")

        # TODO

        # ===========================================================================================
        # Saving model
        logging.info("======================================================================")
        logging.info("Saving Results:")

        # ===========================================================================================
        # Register tracking info
        if env.param.tracking:
            env.publish_results(history=ds.history)
            env.tracking.log_artifacts_folder(local_dir=env.run_folder)

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
        processor = BuildStatic2ValueMain(parameters_file=args.config_file_json)
        processor.run()
    except:
        logging.error("Ops " + str(sys.exc_info()[0]) + " occured!")
        raise
# ===========================================================================================
# ===========================================================================================
