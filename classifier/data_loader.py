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

    def __init__(self, case_path_lst: list, debug: bool=False, test_size: int=20):
        """ Creat a data loader object for the demo cell.

        Args:
            case_path_lst (list): List of use case measurements to use.
            debug (bool): Shows plots of the data if True. 
        """
        self.debug = debug
        self.test_size = test_size
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
        # self._preprocess()
        self._split()
        print('baseline:', 1. - np.sum(self.y_array)/self.y_array.shape[0])


    def write_xy_vectors_to_file(self, path='./input/'):
        pandas.DataFrame(data=self.x_array,
                         columns=['drop_black_pos_y',
                                  'drop_black_pos_z',
                                  'drop_black_pos_x',
                                  'drop_white_pos_x',
                                  'drop_white_pos_y',
                                  'drop_white_pos_z',
                                  'max_belt']).to_csv(path + 'x.csv')
        pandas.DataFrame(data=self.y_array,
                         columns=['quality']).to_csv(path + 'y.csv')

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
            try:
                x, y = self._process_table(sample)        
            except Exception as e:
                print('skipping', sample.use_case, sample.sample_file, e)

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
        grip_lst = []
        pos_lst = []

        def extract_value_and_time(row) -> (np.array, np.array):
            # Extract a value and its time of measurement from a dataframe.
            if row.ServerTimeStamp is np.nan:
                measurement_time = np.datetime64('NaT') # Not a Time
            else:
                measurement_time = datetime.strptime(row.ServerTimeStamp, '%Y-%m-%d %H:%M:%S.%f%z').replace(tzinfo=None)
                measurement_time = np.datetime64(measurement_time)

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
            array[:, 0] = (array[:, 0] - np.mean(array[:, 0]))/(np.std(array[:, 0]) + 0.000001)
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


        def process_belt_data(lst: list) -> np.array:
            # insert a zero placeholder for missing belt data.
            if lst:
                return np.stack(lst)
            else:
                print('Warning belt array empty.',
                      sample.use_case, sample.sample_file)
                return np.zeros((1, 2))
           
        belt1_array = normalize(process_belt_data(conv1_lst))
        belt2_array = normalize(process_belt_data(conv2_lst))
        belt3_array = normalize(process_belt_data(conv3_lst))

        grip_array = np.stack(grip_lst)
        pos_array = np.stack(pos_lst)
        qc_array = np.stack(qc_lst)

        #drop_black_time = pos_array[1, 1]
        #drop_white_time = pos_array[-1, 1]
        drop_black_time = grip_array[1, 1]
        drop_white_time = grip_array[-1, 1]
        


        # compute drop position black
        drop_black_pos_x = np.interp(x=drop_black_time,
                                     xp=x_array[:, 1],
                                     fp=x_array[:, 0])
        drop_black_pos_y = np.interp(x=drop_black_time,
                                     xp=y_array[:, 1],
                                     fp=y_array[:, 0])
        drop_black_pos_z = np.interp(x=drop_black_time,
                                     xp=z_array[:, 1],
                                     fp=z_array[:, 0])
        # compute drop position white
        drop_white_pos_x = np.interp(x=drop_white_time,
                                     xp=x_array[:, 1],
                                     fp=x_array[:, 0])
        drop_white_pos_y = np.interp(x=drop_white_time,
                                     xp=y_array[:, 1],
                                     fp=y_array[:, 0])
        drop_white_pos_z = np.interp(x=drop_white_time,
                                     xp=z_array[:, 1],
                                     fp=z_array[:, 0])

        if self.debug:
            # plot the sample
            plt.title(sample.use_case + '_' + sample.sample_file + '_' + 'arm')
            plt.plot(x_array[:, 1], x_array[:, 0], label='rob x')
            plt.plot(y_array[:, 1], y_array[:, 0], label='rob y')
            plt.plot(z_array[:, 1], z_array[:, 0], label='rob z')
            plt.plot(grip_array[:, 1], grip_array[:, 0], label='grip')
            plt.plot(pos_array[:, 1], pos_array[:, 0], label='pos')
            plt.plot(qc_array[:, 1], qc_array[:, 0], label='qc')
            plt.plot(drop_black_time, drop_black_pos_y, '.', label='drop by')
            plt.plot(drop_black_time, drop_black_pos_z, '.', label='drop bz')
            plt.plot(drop_black_time, drop_black_pos_x, '.', label='drop bx')
            plt.plot(drop_white_time, drop_white_pos_x, '.', label='drop wx')
            plt.plot(drop_white_time, drop_white_pos_y, '.', label='drop wy')
            plt.plot(drop_white_time, drop_white_pos_z, '.', label='drop wz')
            plt.legend()
            plt.show()

            plt.title(sample.use_case + '_' + sample.sample_file + '_' + 'belts')
            plt.plot(belt1_array[:, 1], belt1_array[:, 0], '.', label='belt1')
            plt.plot(belt2_array[:, 1], belt2_array[:, 0], '.', label='belt2')
            plt.plot(belt3_array[:, 1], belt3_array[:, 0], '.', label='belt3')
            plt.show()

            print(sample.use_case)
            print(sample.sample_file)

        # the fastest belt determines the risk of disc slipping.
        max_belt = np.max((np.max(belt1_array[:, 0]),
                           np.max(belt2_array[:, 0]),
                           np.max(belt3_array[:, 0])))
 
        x = np.array([drop_black_pos_y,
                      drop_black_pos_z,
                      drop_black_pos_x,
                      drop_white_pos_x,
                      drop_white_pos_y,
                      drop_white_pos_z,
                      max_belt])
        
        # the last recorded qc value counts.
        y = np.array(qc_array[-1, 0])
        # print(qc_lst)
        return x, y


if __name__ == '__main__':
    path_lst = ['./01_Data/201027/use_case2/Processed/Samples/',
                './01_Data/201027/use_case1/Processed/Samples/']

    # os.chdir(os.path.dirname(__file__))
    # print(os.getcwd())

    # uncommenting this line will show data plots.
    # demo_cell_data = DataLoader(case_path_lst=path_lst, debug=True)

    demo_cell_data = DataLoader(case_path_lst=path_lst, debug=False)

    # uncomment to write new file.
    # demo_cell_data.write_xy_vectors_to_file()
