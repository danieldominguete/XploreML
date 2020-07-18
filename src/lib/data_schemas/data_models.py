from enum import Enum


class StatusOptions(str, Enum):
    to_do = "To Do"
    in_progress = "In Progress"
    done = "Done"
