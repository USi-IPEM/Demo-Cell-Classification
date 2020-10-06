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
        self.x_array = None
        self.y_array = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self._load()
        self._create_xy()
        self._create_data_array()
        self._preprocess()
        self._split()
        print('stop')

    def get_train_xy(self):
        return self.x_train, self.y_train

    def get_test_xy(self):
        return self.x_test, self.y_test

    def _preprocess(self):
        """ Create zero mean and unit std data. """
        x_mean = np.mean(self.x_array, axis=0)
        x_std = np.std(self.x_array, axis=0)
        self.x_array = self.x_array - np.expand_dims(x_mean, 0)
        self.x_array = self.x_array / x_std
        print('mean', np.mean(self.x_array, 0))
        print('std', np.std(self.x_array, 0))

    def _split(self):
        """ Split into train and test data."""
        samples = self.x_array.shape[0]
        self.x_train = self.x_array[:-self.test_size]
        self.x_test = self.x_array[(samples-self.test_size):]
        self.y_train = self.y_array[:-self.test_size]
        self.y_test = self.y_array[(samples-self.test_size):]


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

    def _create_xy(self):
        """ Create the input x and target y for the machine learning
            optimization. """
        for sample in self.raw_sample_list:
            x, y = self._process_table(sample)        
            self.sample_list.append(CellSample(use_case=sample.use_case,
                                               sample_file=sample.sample_file,
                                               raw_data=sample.raw_data,
                                               x=x,
                                               y=y))


    def _process_table(self, sample):
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
                # print(row)
                robot_x_lst.append(float(row.Value))
            if row.PrimaryKey == '528':
                # 528_y: Robot position in y.
                # print(row)
                robot_y_lst.append(float(row.Value))
            if row.PrimaryKey == '529':
                # 529_z: Robot position in z.
                robot_z_lst.append(float(row.Value))
                # print(row)
            if row.PrimaryKey == '27':
                # belt1 data.
                conv1_lst.append(float(row.Value))
            if row.PrimaryKey == '28':
                # belt1 data.
                conv2_lst.append(float(row.Value))
            if row.PrimaryKey == '29':
                # belt1 data.
                conv3_lst.append(float(row.Value))
            if row.PrimaryKey == 'ResultCode':
                qc_lst.append(float(row.Value))
        
        z_val = np.nan
        x_val = np.nan
        y_val = np.nan
        for no, z in enumerate(robot_z_lst):
            if int(z) < 1052000000:
                # drop
                z_val = int(z)
                x_val = int(robot_x_lst[no])
                y_val = int(robot_z_lst[no])
                break
        
        conv1_array = np.array(conv1_lst)
        conv2_array = np.array(conv2_lst)
        conv3_array = np.array(conv3_lst)

        x = np.array([x_val,
                      y_val,
                      z_val,
                      np.max(conv1_lst),
                      np.max(conv2_lst),
                      np.max(conv3_lst)])
        
        if float(qc_lst[-1]) < 1.5:
            qc = 1.
        else:
            qc = 0.
        y = np.array([qc])
        # print(qc_lst)
        return x, y


if __name__ == '__main__':
    path_lst = ['../01_Data/200924/use_case1/Processed/Samples/',
                '../01_Data/200924/use_case2/Processed/Samples/',
                '../01_Data/200924/use_case3/Processed/Samples/',
                '../01_Data/200924/use_case4/Processed/Samples/',
                '../01_Data/200924/use_case5/Processed/Samples/']

    os.chdir(os.path.dirname(__file__))
    print(os.getcwd())

    demo_cell_data = DataLoader(case_path_lst=path_lst, debug=True)
