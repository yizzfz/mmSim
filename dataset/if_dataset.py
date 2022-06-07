from .dataset import *
from simulator import Simulator
from radar import RadarArray
from scene import Point
from util import log, config_to_id, plot_point_cloud_o3d
import motion
import json
import hashlib
import numpy as np
import datetime
import h5py
import multiprocessing
import time

IF_datasets = {
    'human': ['FAUST', 'DynamicFAUST']
}

class IFDataset:
    def __init__(self, name, config, location=None):
        if location is None:
            computer_name = os.environ['COMPUTERNAME']
            location = computer_loc.get(computer_name)
            if location is None:
                raise ValueError
        if not os.path.exists(os.path.join(location, 'if')):
            os.mkdir(os.path.join(location, 'if'))
        self.name = name
        self.id = config_to_id(config, name=name)
        self.fullpath = os.path.join(location, 'if', self.id)
        files = ['train.h5', 'test.h5', 'config.json']
        self.exist = os.path.isdir(self.fullpath)
        for f in files:
            self.exist = self.exist and os.path.isfile(os.path.join(self.fullpath, f))

        self.config = config.copy()
        if self.exist:
            pass
            # print('Existing dataset found')
        else:
            print('New dataset. Awaiting construction')

    def get_path(self):
        return self.fullpath

    def constructed(self):
        return self.exist

    def load(self, train=False):
        if not self.exist:
            raise ValueError('Dataset empty, plase call the construct function')
        filename = os.path.join(self.fullpath, 'train.h5' if train else 'test.h5')
        if not os.path.isfile(filename):
            raise ValueError(f'{filename} not exist')
        f = h5py.File(filename, 'r')
        return f['x'], f['y']

    def construct(self, callback=None, n_trial=10, split=None, thread=8):
        log(f'Constructing {self.name}')
        if not os.path.exists(self.fullpath):
            os.mkdir(self.fullpath)
        layout = self.config['layout']
        ADC_rate = self.config['ADC_rate']
        chirp_time = self.config['chirp_time']
        slope = self.config['slope']
        noise = self.config['noise']
        simulation_period = self.config['simulation_period']
        fps = self.config['fps']
        velocity = self.config['velocity']
        m = getattr(motion, self.config['motion'])(velocity)
        radars = RadarArray((0, 0, 0), layout=layout, ADC_rate=ADC_rate, chirp_time=chirp_time, slope=slope, noise=noise)
        n_rx = len(radars)

        n_train, n_test = 0, 0
        x_shape_batch = (1, 1, n_rx, self.config['steps'], self.config['samples_per_chirp'])
        y_shape_batch = (1, 1, self.config['n_samples'], 3)

        if split is None:
            construct_train = True
            construct_test = True
        elif split == 'train':
            construct_train = True
            construct_test = False
        else:
            construct_train = False
            construct_test = True

        if construct_train:
            n_train = len(Dataset(self.name, n_samples=1, train=True).dataset)
            train_x_shape = (n_trial, n_train, n_rx, self.config['steps'], self.config['samples_per_chirp'])
            train_y_shape = (n_trial, n_train, self.config['n_samples'], 3)
            
        if construct_test:
            n_test = len(Dataset(self.name, n_samples=1, train=False).dataset)
            test_x_shape = (n_trial, n_test, n_rx, self.config['steps'], self.config['samples_per_chirp'])
            test_y_shape = (n_trial, n_test, self.config['n_samples'], 3)

        snr_train = np.zeros((n_trial, n_train))
        snr_test = np.zeros((n_trial, n_test))
        self.total = (n_train + n_test) * n_trial
        self.cnt = 0

        Q = multiprocessing.Manager().Queue()
        pool = multiprocessing.Pool(thread)
        pool.apply_async(self.write_to_file, args=(Q, construct_train, construct_test,  
                                                   x_shape_batch, y_shape_batch,
                                                   train_x_shape,
                                                   train_y_shape,
                                                   test_x_shape,
                                                   test_y_shape, ))

        self.t1 = datetime.datetime.now()
        if construct_train:
            jobs = []
            self.snr_train = snr_train
            for i in range(n_trial):
                train_pc = Dataset(self.name, n_samples=self.config['n_samples'], distance=self.config['distance'], train=True).dataset
                for j in range(n_train):
                    pos = train_pc[j].pos.numpy()
                    scene = [Point(pos, motion=m)]
                    j = pool.apply_async(self.task_wrapper, args=(Q, True, i, j, radars, scene, simulation_period, fps, ),
                                         callback=self.callback_succ, error_callback=self.callback_err)
                    jobs.append(j)
            for j in jobs:
                j.get()
            self.config['snr_train'] = np.mean(self.snr_train)

        if construct_test:
            jobs = []
            self.snr_test = snr_test
            for i in range(n_trial):
                test_pc = Dataset(self.name, n_samples=self.config['n_samples'], distance=self.config['distance'], train=False).dataset
                for j in range(n_test):
                    pos = test_pc[j].pos.numpy()
                    scene = [Point(pos, motion=m)]
                    j = pool.apply_async(self.task_wrapper, args=(Q, False, i, j, radars, scene, simulation_period, fps, ),
                                         callback=self.callback_succ, error_callback=self.callback_err)
                    jobs.append(j)
            for j in jobs:
                j.get()
            self.config['snr_test'] = np.mean(self.snr_test)
        Q.put('kill')
        pool.close()
        pool.join()
        
        with open(os.path.join(self.fullpath, 'config.json'), 'a') as f:
            json.dump(self.config, f, indent=2)

        message = f'{self.name} finished, average train SNR {np.mean(snr_train):.2f} dB, test SNR {np.mean(snr_test):.2f} dB'
        log(message)
        if callback is not None:
            callback(message)

    def task_wrapper(self, Q, isTrain, i, j, radars, scene, simulation_period, fps):
        simulator = Simulator(radars, scene, simulation_period, fps)
        x, snr = simulator.run(ret_snr=True)
        pos = scene[0].get_average_pos()
        Q.put((isTrain, i, j, x, pos))
        return (isTrain, i, j, snr)

    def callback_succ(self, res):
        self.cnt += 1
        (isTrain, i, j, snr) = res
        if isTrain:
            self.snr_train[i, j] = snr
        else:
            self.snr_test[i, j] = snr
        t1 = self.t1
        t2 = datetime.datetime.now()
        eta = ((t2-t1)/(self.cnt)*self.total+t1).strftime('%m.%d-%H:%M')
        print(f'Progress {100*(self.cnt)/self.total:.2f}% [ETA {eta}]', end='\r')

    def callback_err(self, e):
        print(e)

    def write_to_file(self, Q, construct_train, construct_test, 
                      x_shape_batch=None,
                      y_shape_batch=None,
                      train_x_shape=None, 
                      train_y_shape=None, 
                      test_x_shape=None, 
                      test_y_shape=None, 
                      ):
        if construct_train:
            h5fd_train = h5py.File(os.path.join(self.fullpath, 'train.h5'), 'w')
            h5fd_train.create_dataset('x', shape=train_x_shape, chunks=x_shape_batch, dtype=complex)
            h5fd_train.create_dataset('y', shape=train_y_shape, chunks=y_shape_batch, dtype=float)
            train_x = h5fd_train['x']
            train_y = h5fd_train['y']

        if construct_test:
            h5fd_test = h5py.File(os.path.join(self.fullpath, 'test.h5'), 'w')
            h5fd_test.create_dataset('x', shape=test_x_shape, chunks=x_shape_batch, dtype=complex)
            h5fd_test.create_dataset('y', shape=test_y_shape, chunks=y_shape_batch, dtype=float)
            test_x = h5fd_test['x']
            test_y = h5fd_test['y']

        while True:
            if Q.empty():
                time.sleep(10)
                continue
            msg = Q.get()
            if msg == 'kill':
                break
            isTrain, i, j, x, y = msg
            if isTrain:
                train_x[i, j] = x
                train_y[i, j] = y
            else:
                test_x[i, j] = x
                test_y[i, j] = y
        h5fd_train.close()
        h5fd_test.close()
            
