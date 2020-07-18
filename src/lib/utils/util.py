'''
===========================================================================================
Utils Package
===========================================================================================
Script Reviewed by COGNAS
===========================================================================================
'''
import logging, sys
import operator
from functools import reduce
import json
import os


class Util:

    def __init__(self):
        '''Constructor for this class'''

    @classmethod
    def join_lists(cls, l1=None, l2=None, l3=None, l4=None):

        list_columns = l1.copy()
        list_columns.extend(l2)

        if l3 is not None:
            list_columns.extend(l3)

        if l4 is not None:
            list_columns.extend(l4)

        return list_columns

    def print_dataframe_describe(self, dataframe):

        description = dataframe.describe()

        for row in description.iterrows():
            logging.info(row[0])
            for id in row[1].index:
                logging.info('Var: ' + id + ' {a:.3f}'.format(a=(row[1][id])))

        return True

    @classmethod
    def flat_lists(cls, sublist=[]):

        flat_list = reduce(operator.concat, sublist)

        return flat_list

    @staticmethod
    def load_parameters_from_file(path_file: str) -> dict:

        try:
            with open(path_file) as json_file:
                data = json.load(json_file)
        except:
            logging.error('Ops ' + str(sys.exc_info()[0]) + ' occured!')
            raise

        return data

    @staticmethod
    def get_logging_level(level: str) -> int:

        if level == "debug":
            return logging.DEBUG
        elif level == "info":
            return logging.INFO

    @staticmethod
    def get_top_categorical_feature(data, column):
        top = data[column].value_counts().idxmax()
        return top

    @staticmethod
    def get_unique_list(input_list) -> list:
        unique_list = reduce(lambda l, x: l.append(x) or l if x not in l else l, input_list, [])
        return unique_list

    @staticmethod
    def get_nrows_from_file(filepath) -> int:
        f = open(filepath)
        try:
            lines = 1
            buf_size = 1024 * 1024
            read_f = f.read  # loop optimization
            buf = read_f(buf_size)

            # Empty file
            if not buf:
                return 0

            while buf:
                lines += buf.count('\n')
                buf = read_f(buf_size)
        finally:
            f.close()
        return lines

    @staticmethod
    def get_name_and_extension_from_file(filename: str) -> list:
        return os.path.splitext(filename)
