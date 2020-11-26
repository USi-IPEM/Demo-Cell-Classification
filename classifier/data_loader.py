# This module loads the demo cell data.
import os
import pandas
import numpy as np
from datetime import datetime
from random import shuffle, seed
from collections import namedtuple
import matplotlib.pyplot as plt

CellSample = namedtuple('CellSample',
                        ['date',
                         'use_case',
                         'sample_file',
                         'raw_data',
                         'x', 'y'])

class DataLoader(object):
    """ Load the demo-cell data. Base class for vector and sequence loaders. """

    def __init__(self, case_path_lst: list, debug: bool=False,
                 test_size: int=50, seed: int=1, full_y: bool=False,
                 normalize=True):
        """ Creat a data loader object for the demo cell.

        Args:
            case_path_lst (list): List of use case measurements to use.
            debug (bool): Shows plots of the data if True. 
            test_size (int): Number of samples set aside for testing.
            seed (int): Initial seed for the random number generator.
            full_y (bool): If True construct y using qc and distance values.
            normalize (bool): If True data normalization is run.
        """
        self.normalize = normalize
        self.full_y = full_y
        self.seed = seed
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
        self._split()
        if normalize:
            self._normalize()
        if not full_y:
            print('x train shape', self.x_train.shape)
            print('x test shape', self.x_test.shape)
            print('baseline ok   :',
                  '%.1f' % ((1. - np.sum(self.y_array)/self.y_array.shape[0])*100),
                  '%')
            print('baseline fault:',
                  '%.1f' % ((np.sum(self.y_array)/self.y_array.shape[0])*100),
                  '%')

    def _normalize(self):
        """ Normalize train and test data """
        train_mean = np.mean(self.x_train, axis=0)
        train_std = np.std(self.x_train, axis=0)
        self.x_train = (self.x_train - train_mean)/train_std
        self.x_test = (self.x_test - train_mean)/train_std

        if self.full_y:
            # take care of dx,dy and dz.
            yd_train_mean = np.mean(self.y_train[:, 1:], axis=0)
            yd_train_std = np.std(self.y_train[:, 1:], axis=0)
            # normalize only the distances do not touch the labels.
            self.y_train[:, 1:] = (self.y_train[:, 1:] - yd_train_mean) / yd_train_std
            self.y_test[:, 1:] = (self.y_test[:, 1:] - yd_train_mean) / yd_train_std

    def get_train_xy(self):
        """ Returns the training data vectors.

        Returns:
            x [np.array]: Input vectors.
            y [np.array]: Target vectors.
        """        
        return self.x_train, self.y_train

    def get_test_xy(self):
        """ Returns the test data vectors.

        Returns:
            x [np.array]: Input vectors.
            y [np.array]: Target vectors.
        """        
        return self.x_test, self.y_test

    def _split(self):
        """ Split into train and test data."""
        samples = self.x_array.shape[0]
        self.x_train = self.x_array[:-self.test_size]
        self.x_test = self.x_array[(samples-self.test_size):]
        self.y_train = self.y_array[:-self.test_size]
        self.y_test = self.y_array[(samples-self.test_size):]

    def _create_data_array(self):
        """ Create and shuffle an array with all samples. """
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
        # Seed the numpy generator to ensure reproducible results.
        seed(self.seed)
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
                    date = path.split('/')[2]
                    use_case = path.split('/')[3]
                    sample_file = current_file
                    sample_list.append(CellSample(use_case=use_case,
                                                  date=date,
                                                  sample_file=sample_file,
                                                  raw_data=pandas_file,
                                                  x=None, y=None))
        self.raw_sample_list = sample_list

    def _create_xy(self):
        """ Create the input x and target y for the machine learning
            optimization. """
        for sample in self.raw_sample_list:
            try:
                x, y = self._process_table(sample, self.full_y)   
                self.sample_list.append(CellSample(use_case=sample.use_case,
                                                sample_file=sample.sample_file,
                                                date=sample.date,
                                                raw_data=sample.raw_data,
                                                x=x,
                                                y=y))
            except Exception as e:
                print('skipping', sample.use_case, sample.sample_file, e)

    def _load_table(self, sample):
        """ Extrat the input x and target y values from the current
            data frame"""
 
        raw_data = sample.raw_data
        rows, cols = raw_data.shape
        # create a list for each value we are parsing.
        robot_x_lst = []
        robot_y_lst = []
        robot_z_lst = []
        conv1_lst = []
        conv2_lst = []
        conv3_lst = []
        qc_lst = []
        dx_lst = []
        dy_lst = []
        da_lst = []
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
            if row.PrimaryKey == 'DistanceX':
                dx_lst.append(extract_value_and_time(row))
            if row.PrimaryKey == 'DistanceY':
                dy_lst.append(extract_value_and_time(row))
            if row.PrimaryKey == 'DistanceAbs':
                da_lst.append(extract_value_and_time(row))
            if row.PrimaryKey == '561':
                # Position indicator
                grip_lst.append(extract_value_and_time(row))
            if row.PrimaryKey == '560':
                # Grip indicator.
                pos_lst.append(extract_value_and_time(row))

        return robot_x_lst, robot_y_lst, robot_z_lst, \
               conv1_lst, conv2_lst, conv3_lst, \
               qc_lst, dx_lst, dy_lst, da_lst, \
               grip_lst, pos_lst

    def _process_table(self, sample):
        # Implemented in Child classes Vector
        # or Sequence Loader.
        raise NotImplementedError

    def _process_belt_data(self, lst: list, sample) -> np.array:
        # insert a zero placeholder for missing belt data.
        if lst:
            return np.stack(lst)
        else:
            print('Warning belt array empty.',
                  sample.date, sample.use_case, sample.sample_file)
            return np.zeros((1, 2))


class VectorLoader(DataLoader):
    """ Load x,y vector pairs for use with scikit learn models. """

    def _process_table(self, sample, full_y=False):
        """ Process tabulated input data and return the model input
            and target values.

        Args:
            sample (CellSample): Cell sample encoding a processed 
                input file.

        Returns:
            x: (np.array) model input.
            y: (np.array) model output.
        """
        robot_x_lst, robot_y_lst, robot_z_lst, \
            conv1_lst, conv2_lst, conv3_lst, \
            qc_lst, dx_lst, dy_lst, da_lst, \
            grip_lst, pos_lst = self._load_table(sample)

        x_array = np.stack(robot_x_lst)
        y_array = np.stack(robot_y_lst)
        z_array = np.stack(robot_z_lst)

        belt1_array = self._process_belt_data(conv1_lst, sample)
        belt2_array = self._process_belt_data(conv2_lst, sample)
        belt3_array = self._process_belt_data(conv3_lst, sample)

        grip_array = np.stack(grip_lst)
        pos_array = np.stack(pos_lst)

        qc_array = np.stack(qc_lst)
        dx_array = np.stack(dx_lst)
        dy_array = np.stack(dy_lst)
        da_array = np.stack(da_lst)

        #drop_white_time = pos_array[1, 1]
        #drop_black_time = pos_array[-1, 1]
        drop_white_time = grip_array[1, 1]
        drop_black_time = grip_array[-1, 1]
        
        # compute drop position black
        drop_white_pos_x = np.interp(x=drop_white_time,
                                     xp=x_array[:, 1],
                                     fp=x_array[:, 0])
        drop_white_pos_y = np.interp(x=drop_white_time,
                                     xp=y_array[:, 1],
                                     fp=y_array[:, 0])
        drop_white_pos_z = np.interp(x=drop_white_time,
                                     xp=z_array[:, 1],
                                     fp=z_array[:, 0])
        # compute drop position white
        drop_black_pos_x = np.interp(x=drop_black_time,
                                     xp=x_array[:, 1],
                                     fp=x_array[:, 0])
        drop_black_pos_y = np.interp(x=drop_black_time,
                                     xp=y_array[:, 1],
                                     fp=y_array[:, 0])
        drop_black_pos_z = np.interp(x=drop_black_time,
                                     xp=z_array[:, 1],
                                     fp=z_array[:, 0])

        if self.debug:
            # plot the sample
            x_mean = np.mean(x_array[:, 1])
            x_std = np.std(x_array[:, 1])
            plt.title(sample.use_case + '_' + sample.sample_file + '_' + 'arm')
            plt.plot((x_array[:, 1]-x_mean)/x_std, x_array[:, 0], label='rob x')
            plt.plot((y_array[:, 1]-x_mean)/x_std, y_array[:, 0], label='rob y')
            plt.plot((z_array[:, 1]-x_mean)/x_std, z_array[:, 0], label='rob z')
            plt.plot((grip_array[:, 1]-x_mean)/x_std, grip_array[:, 0], label='grip')
            plt.plot((pos_array[:, 1]-x_mean)/x_std, pos_array[:, 0], label='pos')
            plt.plot((qc_array[:, 1]-x_mean)/x_std, qc_array[:, 0], label='qc')
            plt.plot((drop_black_time-x_mean)/x_std, drop_black_pos_x, '.', label='drop bx')
            plt.plot((drop_black_time-x_mean)/x_std, drop_black_pos_y, '.', label='drop by')
            plt.plot((drop_black_time-x_mean)/x_std, drop_black_pos_z, '.', label='drop bz')
            plt.plot((drop_white_time-x_mean)/x_std, drop_white_pos_x, '.', label='drop wx')
            plt.plot((drop_white_time-x_mean)/x_std, drop_white_pos_y, '.', label='drop wy')
            plt.plot((drop_white_time-x_mean)/x_std, drop_white_pos_z, '.', label='drop wz')
            plt.legend()
            plt.show()

            plt.title(sample.use_case + '_' + sample.sample_file + '_' + 'belts')
            plt.plot(belt1_array[:, 1], belt1_array[:, 0], '.', label='belt1')
            plt.plot(belt2_array[:, 1], belt2_array[:, 0], '.', label='belt2')
            plt.plot(belt3_array[:, 1], belt3_array[:, 0], '.', label='belt3')
            plt.show()

            print(sample.date)
            print(sample.use_case)
            print(sample.sample_file)

        # the fastest belt determines the risk of disc slipping.
        max_belt = np.max((np.max(belt1_array[:, 0]),
                           np.max(belt2_array[:, 0]),
                           np.max(belt3_array[:, 0])))
 
        x = np.array([drop_white_pos_y,
                      drop_white_pos_z,
                      drop_white_pos_x,
                      drop_black_pos_x,
                      drop_black_pos_y,
                      drop_black_pos_z,
                      max_belt])
        
        # the last recorded qc value counts.

        if full_y:
            y = np.array([qc_array[-1, 0],
                          dx_array[-1, 0],
                          dy_array[-1, 0],
                          da_array[-1, 0]])
        else:
            y = np.array(qc_array[-1, 0])

        # print(qc_lst)
        return x, y

    def write_xy_vectors_to_file(self, path='./input/'):
        """Write the svm input into a csv file.
        Args:
            path (str, optional): [description]. Defaults to './input/'.
        """
        def write_x(x_array, file_name='x.csv'):
            pandas.DataFrame(data=x_array,
                            columns=['drop_white_pos_y',
                                    'drop_white_pos_z',
                                    'drop_white_pos_x',
                                    'drop_black_pos_x',
                                    'drop_black_pos_y',
                                    'drop_black_pos_z',
                                    'max_belt']).to_csv(path + file_name)
        
        def write_y(y_array, file_name='y.csv'):
            pandas.DataFrame(data=y_array,
                            columns=['quality',
                                    'dx',
                                    'dy',
                                    'da']).to_csv(path + file_name)


        write_x(self.x_array, 'x_all.csv')
        write_x(self.x_train, 'x_train.csv')
        write_x(self.x_test, 'x_test.csv')

        write_y(self.y_array, 'y_all.csv')
        write_y(self.y_train, 'y_train.csv')
        write_y(self.y_test,  'y_test.csv')

        all_in_one = np.concatenate([self.x_array, self.y_array], axis=1)
        pandas.DataFrame(data=all_in_one,
                         columns=['drop_white_pos_y',
                                  'drop_white_pos_z',
                                  'drop_white_pos_x',
                                  'drop_black_pos_x',
                                  'drop_black_pos_y',
                                  'drop_black_pos_z',
                                  'max_belt',
                                  'quality',
                                  'dx',
                                  'dy',
                                  'da']).to_csv(path + 'all_in_one.csv')
        print('done')



if __name__ == '__main__':
    # path_lst = ['./01_Data/201027/use_case2/Processed/Samples/',
    #             './01_Data/201027/use_case1/Processed/Samples/']

    path_lst = ['./01_Data/201027/use_case2/Processed/Samples/',
                './01_Data/201027/use_case1/Processed/Samples/',
                './01_Data/201030/use_case1/Processed/Samples/',
                './01_Data/201030/use_case2/Processed/Samples/',
                './01_Data/201030/use_case3/Processed/Samples/',
                './01_Data/201030/use_case4/Processed/Samples/',
                './01_Data/201030/use_case5/Processed/Samples/',
                './01_Data/201030/use_case6/Processed/Samples/']

    # os.chdir(os.path.dirname(__file__))
    # print(os.getcwd())

    # uncommenting this line will show data plots.
    # demo_cell_data = VectorLoader(case_path_lst=path_lst, debug=True)
    # sequence_data = SequenceLoader(case_path_lst=path_lst)

    demo_cell_data = VectorLoader(case_path_lst=path_lst, debug=False,
                                  full_y=True, normalize=True)

    # uncomment to write new file.
    demo_cell_data.write_xy_vectors_to_file()
