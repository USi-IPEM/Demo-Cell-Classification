import os
import pandas
import numpy as np
from datetime import datetime
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
        grip_lst = []
        pos_lst = []

        def extract_value_and_time(row) -> (np.array, np.array):
            # Extract a value and its time of measurement from a dataframe.
            if row.ServerTimeStamp is np.nan:
                measurement_time = np.datetime64('NaT') # Not a Time
            else:
                measurement_time = np.datetime64(datetime.strptime(row.ServerTimeStamp, '%Y-%m-%d %H:%M:%S.%f%z'))

            if row.Value == 'true':
                measurement_value = 1.0
            elif row.Value == 'false':
                measurement_value = 0.0
            else:
                measurement_value = np.array(float(row.Value))

            return (measurement_value,
                    measurement_time.astype("float"))

        def normalize(array: np.array) -> np.array:
            # normalize the input array first channel.
            array[:, 0] = (array[:, 0] - np.mean(array[:, 0]))/np.std(array[:, 0])
            return array

        for row in raw_data.itertuples():
            # current_row = raw_data[row_no, :]
            if row.PrimaryKey == '527':
                # 527_x: Robot position in x.
                robot_x_lst.append(extract_value_and_time(row))
            if row.PrimaryKey == '528':
                # 528_y: Robot position in y.
                robot_y_lst.append(extract_value_and_time(row))
            if row.PrimaryKey == '529':
                # 529_z: Robot position in z.
                robot_z_lst.append(extract_value_and_time(row))
            if row.PrimaryKey == '27':
                # belt1 data.
                conv1_lst.append(extract_value_and_time(row))
            if row.PrimaryKey == '28':
                # belt2 data.
                conv2_lst.append(extract_value_and_time(row))
            if row.PrimaryKey == '29':
                # belt3 data.
                conv3_lst.append(extract_value_and_time(row))
            if row.PrimaryKey == 'ResultCode':
                qc_lst.append(extract_value_and_time(row))
            if row.PrimaryKey == '561':
                # Position indicator
                grip_lst.append(extract_value_and_time(row))
            if row.PrimaryKey == '560':
                # Grip indicator.
                pos_lst.append(extract_value_and_time(row))

        x_array = normalize(np.stack(robot_x_lst))
        y_array = normalize(np.stack(robot_y_lst))
        z_array = normalize(np.stack(robot_z_lst))

        belt1_array = np.stack(conv1_lst)
        belt2_array = np.stack(conv2_lst)
        belt3_array = np.stack(conv3_lst)

        grip_array = np.stack(grip_lst)
        pos_array = np.stack(pos_lst)
        qc_array = np.stack(qc_lst)

        plt.title(sample.use_case + '_' + sample.sample_file)
        plt.plot(x_array[:, 1], x_array[:, 0], label='rob x')
        plt.plot(y_array[:, 1], y_array[:, 0], label='rob y')
        plt.plot(z_array[:, 1], z_array[:, 0], label='rob z')
        plt.plot(grip_array[:, 1], grip_array[:, 0], label='grip')
        plt.plot(pos_array[:, 1], pos_array[:, 0], label='pos')
        plt.plot(qc_array[:, 1], qc_array[:, 0], label='qc')
        plt.plot()
        plt.legend()
        plt.show()

        # plot the sample
        print(sample.use_case)
        print(sample.sample_file)
 
 
 
 
if __name__ == '__main__':
    path_lst = ['./01_Data/201027/use_case1/Processed/Samples/',
                ]

    # os.chdir(os.path.dirname(__file__))
    # print(os.getcwd())

    demo_cell_data = DataLoader(case_path_lst=path_lst, debug=True)
