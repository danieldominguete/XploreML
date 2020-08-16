"""
===========================================================================================
Txt2Vec : Language Model building for vectorize txt features
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
from src.lib.data_schemas.txt2vec_parameters import Txt2VecParameters
from src.lib.data_schemas.environment_parameters import EnvironmentParameters
from src.lib.environment.environment import Environment
from src.lib.data_processing.data_processing import DataProcessing
from src.lib.data_schemas.word2vec_parameters import XWord2VecParameters
from src.lib.nlp.word2vec import XWord2Vec



class BuildTxt2VecMain:
    def __init__(self, parameters_file):

        """Constructor for this class"""
        self.parameters_file = parameters_file

    def model_selection(self, data_config, data_param, environment):

        # Validade and load model hyperparameters
        if data_param.model_type == "word2vec":
            model_param = XWord2VecParameters(
                **data_config.get("word2vec_parameters")
            )
            model = XWord2Vec(param=model_param, application=data_param.application, application_type=data_param.application_type, env=environment)
        else:
            logging.error('Language model type not valid.')

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
        data_param = Txt2VecParameters(**data_config.get("txt2vec_parameters"))
        ds = DataProcessing(param=data_param)

        # ===========================================================================================
        # Setup environment
        env.init_script(script_name=os.path.basename(__file__), warnings_level=PYTHON_WARNINGS)

        # ===========================================================================================
        # Loading data
        logging.info("======================================================================")
        logging.info("Loading Training and Test Data:")
        data_train_input, data_train_target = ds.load_dataset()

        logging.info("======================================================================")
        logging.info("Preprocessing Training Data:")
        (
            data_train_input,
            variables_input
        ) = ds.prepare_corpus_data(data=data_train_input)

        logging.info("======================================================================")
        logging.info("Building Model:")

        # select model technology
        model = self.model_selection(data_config=data_config, data_param=data_param, environment=env)

        # build model
        model.fit(
            dataframe= data_train_input,
            corpus_col=variables_input[0]
        )

        logging.info("======================================================================")
        logging.info("Training Results")

        # todo a embedding evaluation
        # model_eval_train = ClassificationModelEvaluation(
        #     Y_target=data_train_target[data_param.output_target],
        #     Y_predict=data_train_predict[['predict']],
        #     subset_label="Train",
        #     classification_type=data_param.classification_type,
        #     Y_int_to_cat_labels=int_to_cat_dict_list_target,
        #     Y_cat_to_int_labels=cat_to_int_dict_list_target,
        #     history=None,
        # )

        #model_eval_train.print_evaluation_scores()
        #env.tracking.publish_c_eval(model_eval=model_eval_train, mode="train")

        if env_param.view_plots or env_param.save_plots:
            logging.info("======================================================================")
            logging.info("Plotting training result graphs")

            if env_param.save_plots:
                logging.info("Plots will save in " + env.run_folder)

            if env_param.view_plots:
                logging.info("Plots will view in window popup")

            # model_eval_train.plot_evaluation_scores(
            #     view=env_param.view_plots,
            #     save=env_param.save_plots,
            #     path=env.run_folder,
            #     prefix=env.prefix_name + "train_",
            # )

        # ===========================================================================================
        # Saving model
        logging.info("======================================================================")
        logging.info("Saving Results:")

        model.save_model()

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
        processor = BuildTxt2VecMain(parameters_file=args.config_file_json)
        processor.run()
    except:
        logging.error("Ops " + str(sys.exc_info()[0]) + " occured!")
        raise
# ===========================================================================================
# ===========================================================================================
