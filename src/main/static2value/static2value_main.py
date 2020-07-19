"""
===========================================================================================
Static2Value : Model building from static data to value prediction
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
from src.lib.data_schemas.static2value_parameters import Static2ValueParameters
from src.lib.data_schemas.environment_parameters import EnvironmentParameters
from src.lib.data_processing.data_processing import DataProcessing
from src.lib.environment.environment import Environment
from src.lib.data_schemas.linear_regression_parameters import LinearRegressionParameters
from src.lib.model.linear_regression import XLinearRegression
from src.lib.model.model_evaluation import RegressionModelEvaluation


class BuildStatic2ValueMain:
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
        data_param = Static2ValueParameters(**data_config.get("static2value_parameters"))
        ds = DataProcessing(param=data_param)

        # Validade and load model hyperparameters
        if data_param.model_type == "linear_regression":
            model_param = LinearRegressionParameters(
                **data_config.get("linear_regression_parameters")
            )
            model = XLinearRegression(model_param)

        # ===========================================================================================
        # Setup environment
        env.init_script(script_name=os.path.basename(__file__), warnings_level=PYTHON_WARNINGS)

        # ===========================================================================================
        # Loading data
        logging.info("======================================================================")
        logging.info("Loading Training and Test Data:")
        data_train_input, data_train_target = ds.load_train_data()
        data_test_input, data_test_target = ds.load_test_data()

        logging.info("======================================================================")
        logging.info("Preprocessing Training Data:")
        (
            data_train_input,
            data_train_target,
            data_test_input,
            data_test_target,
            variables_input,
            variables_target,
        ) = ds.prepare_train_test_data(data_train_input=data_train_input, data_train_target=data_train_target, data_test_input=data_test_input, data_test_target=data_test_target)

        logging.info("======================================================================")
        logging.info("Building Model:")
        model.fit(
            data_input=data_train_input[variables_input],
            data_target=data_train_target[variables_target],
        )

        logging.info("======================================================================")
        logging.info("Building predictions:")

        data_train_predict = model.eval_predict(data_input=data_train_input[variables_input])

        logging.info("======================================================================")
        logging.info("Training Results")

        model_eval_train = RegressionModelEvaluation(
            Y_target=data_train_target[variables_target],
            Y_predict=data_train_predict,
            subset_label="Train",
            history=None,
        )

        model_eval_train.print_evaluation_scores()
        env.tracking.publish_regression_eval(model_eval=model_eval_train, mode='train')

        if env_param.view_plots or env_param.save_plots:
            logging.info('Plotting training result graphs')
            model_eval_train.plot_evaluation_scores(view=env_param.view_plots,
                                                    save=env_param.save_plots,
                                                    path=env.run_folder,
                                                    prefix=env.prefix_name + 'train_')


        logging.info("======================================================================")
        logging.info("Test Results")
        data_test_predict = model.eval_predict(data_input=data_test_input[variables_input])

        model_eval_test = RegressionModelEvaluation(
            Y_target=data_test_target[variables_target],
            Y_predict=data_test_predict,
            subset_label="Test",
            history=None,
        )

        model_eval_test.print_evaluation_scores()
        env.tracking.publish_regression_eval(model_eval=model_eval_test, mode='test')

        if env_param.view_plots or env_param.save_plots:
            logging.info('Plotting test result graphs')
            model_eval_test.plot_evaluation_scores(view=env_param.view_plots,
                                                   save=env_param.save_plots,
                                                    path = env.run_folder,
                                                    prefix = env.prefix_name + 'test_')

        # ===========================================================================================
        # Saving model
        logging.info("======================================================================")
        logging.info("Saving Datasets:")

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
