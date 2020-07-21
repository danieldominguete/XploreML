"""
===========================================================================================
Environment Parameters Class
===========================================================================================
"""

from pydantic import BaseModel
from datetime import datetime, date, time
from enum import Enum
from typing import Optional
import logging


class EnvironmentId(str, Enum):
    console_localhost = 'console_localhost'


class LoggingLevel(str, Enum):
    '''
        "debug": Detailed information, for diagnosing problems. Value=10.
        "info": Confirm things are working as expected. Value=20.
        "warnning": Something unexpected happened, or indicative of some problem. But the software is still working as expected. Value=30.
        "error": More serious problem, the software is not able to perform some function. Value=40
        "critical": A serious error, the program itself may be unable to continue running. Value=50
    '''
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EnvironmentParameters(BaseModel):

    # prefix for output files
    app_name: str

    # environment identification
    environment_id: EnvironmentId

    # path to working folder
    output_path: str = "outputs"

    # verbose level for logging
    logging_level: LoggingLevel

    # Activate show plot images
    view_plots: bool

    # Activate save plot images
    save_plots: bool

    # Activate mlflow register
    tracking: bool
