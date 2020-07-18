'''
===========================================================================================
Environment Package
===========================================================================================
Script by COGNAS
===========================================================================================
'''
import logging
import datetime, time, os
import warnings
from dotenv import load_dotenv, find_dotenv
from src.lib.utils.util import Util
from src.lib.environment.mlflow_management import MLFlowManagement

class Environment:

    def __init__(self, param=None):
        """Constructor for this class"""
        self.param = param
        self.app_folder = None
        self.run_folder = None
        self.prefix_name = None
        self.output_path = None
        self.script_name = None
        self.time_init = None
        self.datetime_init = None
        self.prefix_name = ""
        self.tracking=None

        # =========================================================================================
        # Environment variables
        load_dotenv(find_dotenv())
        self.MLFLOW_URI = os.getenv("MLFLOW_URI")


    def init_script(self, script_name:str="", warnings_level:str="") -> bool:

        if self.param.environment_id == 'console_localhost':

            # Measure time consuption
            self.datetime_init = datetime.datetime.now()
            self.time_init = time.time()
            self.script_name = script_name

            log = Util.get_logging_level(level=self.param.logging_level)
            logging.basicConfig(level=log,
                                format='%(asctime)s: %(levelname)s - %(message)s',
                                datefmt='%y-%m-%d %H:%M')

            # Supress warnings
            warnings.filterwarnings(warnings_level)

            logging.info("======================================================================")
            logging.info('Starting ' + self.script_name + ' at ' + str(self.datetime_init.strftime("%Y-%m-%d %H:%M")))

            # setup working files
            self.set_run_identificaton(script_name=self.script_name)

            # create app folder
            if not os.path.exists(self.app_folder):
                os.makedirs(self.app_folder)

            # create run folder
            if not os.path.exists(self.run_folder):
                os.makedirs(self.run_folder)

            # setting tracking server
            if self.param.tracking:
                self.tracking = MLFlowManagement()
                tag = self.datetime_init.strftime("%y-%m-%d-%H-%M-%S")
                self.tracking.setup_mlflow_tracking(URI=self.MLFLOW_URI, experiment_name=self.param.app_name, run_name=tag)

        else:
            logging.error('Environment id not valid')

        return True

    def close_script(self):
        time_ref_end = datetime.datetime.now()
        time_end = time.time()
        exec_time = (((time_end - self.time_init) / 60) / 60)

        logging.info("======================================================================")
        logging.info('Conclusion at ' + str(time_ref_end.strftime("%Y-%m-%d %H:%M")))
        logging.info('Execution time: %.2f hours' % exec_time)
        logging.info("======================================================================")
        logging.info("                                                             cognas.ai")
        logging.info("======================================================================")

        return True

    def set_run_identificaton(self, script_name:str) -> bool:

        #get only filename
        script_name, _ = Util.get_name_and_extension_from_file(script_name)

        tag = self.datetime_init.strftime("%y-%m-%d-%H-%M-%S")
        self.app_folder = self.param.output_path + "/" + self.param.app_name + "/"
        self.run_folder = self.param.output_path + "/" + self.param.app_name + "/" + script_name + "_" + tag + "/"
        self.prefix_name = self.param.app_name + "_" + tag + "_"
        return True

    def publish_results(self, history):

        #mlflow tracking
        self.tracking.publish_history(history)
        return True
