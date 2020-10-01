import os
import pandas
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt

CellSample = namedtuple('CellSample',
                        ['use_case',
                         'sample_file',
                         'raw_data',
                         'x', 'y'])

class DataLoader(object):
    """ Load the demo-cell data. """

    def __init__(self, case_path_lst: list, debug: bool):
        """ Creat a data loader object for the demo cell.

        Args:
            case_path_lst (list): List of use case measurements to use.
            debug (bool): Shows plots of the data if True. 
        """
        self.case_path_lst = case_path_lst
        self.raw_sample_list = None
        self.sample_list = None
        self._load()
        self._create_xy()

    def _load(self):
        """ Load the processed csv files and list their values as
            pandas data frames in raw sample list. """
        sample_list = []
        for path in self.case_path_lst:
            for root, dirs, files in os.walk(path):
                for current_file in files:
                    pandas_file = pandas.read_csv(root + current_file)
                    use_case = path.split('/')[3]
                    sample_file = current_file
                    sample_list.append(CellSample(use_case=use_case,
                                                  sample_file=sample_file,
                                                  raw_data=pandas_file,
                                                  x=None, y=None))
        self.raw_sample_list = sample_list

    def _create_xy(self):
        """ Create the input x and target y for the machine learning
            optimization. """
        for sample in self.raw_sample_list:
            x, y = self._process_table(sample.raw_data)        
            self.sample_list.append(CellSample(use_case=sample.use_case,
                                               sample_file=sample.sample_file,
                                               raw_data=sample.raw_data,
                                               x=x,
                                               y=y))


    def _process_table(self, raw_data: pandas.DataFrame):
        """ Extrat the input x and target y values from the current
            data frame"""
        rows, cols = raw_data.shape
        robot_x_lst = []
        robot_y_lst = []
        robot_z_lst = []
        belt_max_lst = []  # belt values are missing.
        qc_lst = []
        for row in raw_data.itertuples():
            # current_row = raw_data[row_no, :]
            if row.PrimaryKey == '527':
                # 527_x: Robot position in x.
                print(row)
                robot_x_lst.append(row.Value)
            if row.PrimaryKey == '528':
                # 528_y: Robot position in y.
                print(row)
                robot_y_lst.append(row.Value)
            if row.PrimaryKey == '529':
                # 529_z: Robot position in z.
                robot_z_lst.append(row.Value)
                print(row)
            if row.PrimaryKey == 'conv':
                # placeholder for belt data.
                pass
            if row.PrimaryKey == 'ResultCode':
                qc_lst.append(row.Value)
        
        plt.Figure()
        plt.plot(robot_x_lst)
        plt.plot(robot_y_lst)
        plt.plot(robot_z_lst)
        plt.show()
        plt.Figure()
        plt.plot(qc_lst)
        plt.show()
        print('stop')


if __name__ == '__main__':
    path_lst = ['./01_Data/200924/use_case1/Processed/Samples/',
                './01_Data/200924/use_case2/Processed/Samples/',
                './01_Data/200924/use_case3/Processed/Samples/',
                './01_Data/200924/use_case4/Processed/Samples/',
                './01_Data/200924/use_case5/Processed/Samples/']

    demo_cell_data = DataLoader(case_path_lst=path_lst, debug=True)
