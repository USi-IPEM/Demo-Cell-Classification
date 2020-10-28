import os
import pandas
import numpy as np
from random import shuffle
from collections import namedtuple
import matplotlib.pyplot as plt

CellSample = namedtuple('CellSample',
                        ['use_case',
                         'sample_file',
                         'raw_data',
                         'x', 'y'])


class DataLoader(object):
    """ Load the demo-cell data. """

    def __init__(self, case_path_lst: list, debug: bool, test_size=20):
        """ Creat a data loader object for the demo cell.

        Args:
            case_path_lst (list): List of use case measurements to use.
            debug (bool): Shows plots of the data if True. 
        """
        self.test_size = 20
        self.case_path_lst = case_path_lst
        self.raw_sample_list = []
        self.sample_list = []
        self._load()
        self._plot()
        print('stop')

    def _create_data_array(self):
        """ Create and shuffle an array with all samples.
        """
        x = []
        y = []
        for sample in self.sample_list:
            if not np.isnan(np.sum(sample.x)):
                x.append(sample.x)
                y.append(sample.y)
            else:
                print('skipping', sample.use_case, sample.sample_file, 
                      'no valid z value for drop.')

        x_rnd = []
        y_rnd = []
        index_rnd = list(range(len(x)))
        shuffle(index_rnd)
        for i in index_rnd:
            x_rnd.append(x[i])
            y_rnd.append(y[i])

        self.x_array = np.stack(x_rnd, axis=0)
        self.y_array = np.stack(y_rnd, axis=0)
        assert self.x_array.shape[0] == self.y_array.shape[0]

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

    def _plot(self):
        """ Create the input x and target y for the machine learning
            optimization. """
        for sample in self.raw_sample_list:
            self._plot_table(sample)        
  
 
    def _plot_table(self, sample):
        """ Extrat the input x and target y values from the current
            data frame"""
        raw_data = sample.raw_data
        rows, cols = raw_data.shape
        robot_x_lst = []
        robot_y_lst = []
        robot_z_lst = []
        conv1_lst = []
        conv2_lst = []
        conv3_lst = []
        qc_lst = []
        for row in raw_data.itertuples():
            # current_row = raw_data[row_no, :]
            if row.PrimaryKey == '527':
                # 527_x: Robot position in x.
                robot_x_lst.append(float(row.Value))
            if row.PrimaryKey == '528':
                # 528_y: Robot position in y.
                robot_y_lst.append(float(row.Value))
            if row.PrimaryKey == '529':
                # 529_z: Robot position in z.
                robot_z_lst.append(float(row.Value))
            if row.PrimaryKey == '27':
                # belt1 data.
                conv1_lst.append(float(row.Value))
            if row.PrimaryKey == '28':
                # belt2 data.
                conv2_lst.append(float(row.Value))
            if row.PrimaryKey == '29':
                # belt3 data.
                conv3_lst.append(float(row.Value))
            if row.PrimaryKey == 'ResultCode':
                qc_lst.append(float(row.Value))
        
        # plot the sample
        print(sample.use_case)
        print(sample.sample_file)
        plt.title(sample.use_case + '_' + sample.sample_file)
        plt.plot(robot_x_lst, label='x')
        plt.plot(robot_y_lst, label='y')
        plt.plot(robot_z_lst, label='z')
        plt.legend()
        plt.show()
 
        input('press any key to continue.')
 
 
 
if __name__ == '__main__':
    path_lst = ['../01_Data/201027/use_case1/Processed/Samples/',
                ]

    # os.chdir(os.path.dirname(__file__))
    # print(os.getcwd())

    demo_cell_data = DataLoader(case_path_lst=path_lst, debug=True)
