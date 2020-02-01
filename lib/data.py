import os
import csv
import glob
import numpy as np
from collections import OrderedDict
from lib import indicators
from gym.utils import seeding
import torch

# Prices = collections.namedtuple('Prices', field_names=['open', 'high', 'low', 'close', 'volume'])

def float_available(f):
    if f == "null":
        return float(0)
    else:
        return float(f)

class csv_reader:
    def __init__(self):
        self.total_count_filter = 0
        self.total_count_out = 0
        self.total_count_fixed = 0

    def read_csv(self, file_name, sep='\t', filter_data=True, fix_open_price=False):
        data = {}
        print("Reading", file_name)
        with open(file_name, 'rt', encoding='utf-8') as fd:
            reader = csv.reader(fd, delimiter=sep)
            h = next(reader)
            if '<OPEN>' not in h and sep == ',':
                return self.read_csv(file_name, ';')
            indices = [h.index(s) for s in ('<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>')]
            o, h, l, c, v = [], [], [], [], []
            count_out = 0
            count_filter = 0
            count_fixed = 0
            prev_vals = None
            for row in reader:
                vals = list(map(float_available, [row[idx] for idx in indices]))
                if filter_data and ((vals[-1] < (1e-8))): # filter out the day when no volume
                    count_filter += 1
                    continue

                po, ph, pl, pc, pv = vals

                # fix open price for current bar to match close price for the previous bar
                if fix_open_price and prev_vals is not None:
                    ppo, pph, ppl, ppc, ppv = prev_vals
                    if abs(po - ppc) > 1e-8:
                        count_fixed += 1
                        po = ppc
                        pl = min(pl, po)
                        ph = max(ph, po)
                count_out += 1
                o.append(po)
                c.append(pc)
                h.append(ph)
                l.append(pl)
                v.append(pv)
                prev_vals = vals
        print("Read done, got %d rows, %d filtered, %d open prices adjusted" % (
            count_filter + count_out, count_filter, count_fixed))
        # stored data
        self.total_count_filter += count_filter
        self.total_count_out += count_out
        self.total_count_fixed += count_fixed
        # stacking
        data['open'] = np.array(o, dtype=np.float64)
        data['high'] = np.array(h, dtype=np.float64)
        data['low'] = np.array(l, dtype=np.float64)
        data['close'] = np.array(c, dtype=np.float64)
        data['volume'] = np.array(v, dtype=np.float64)
        return data

class SimpleSpliter:
    def __init__(self):
        self.trainSet_size = 0
        self.testSet_size = 0
        self.offset = 0

    def split_data(self, data, percentage=0.8):
        assert (isinstance(data, dict))
        train_data = {}
        test_data = {}
        self.offset = np.int(data['close'].shape[0] * percentage)
        for key in list(data.keys()):
            train_data[key] = data[key][:self.offset]
            test_data[key] = data[key][self.offset:]

        print("Split data done, training data: %d rows, eval data: %d" %
              (train_data['close'].shape[0], test_data['close'].shape[0]))
        self.trainSet_size += train_data['close'].shape[0]
        self.testSet_size += test_data['close'].shape[0]
        return train_data, test_data

def price_files(dir_name):
    result = []
    for path in glob.glob(os.path.join(dir_name, "*.csv")):
        result.append(path)
    return result

def load_fileList(path):
    file_list = os.listdir(path)
    return file_list, path

def addition_indicators(prices, trend_names, status_names):
    trend_indicators = OrderedDict()
    status_indicators = OrderedDict()
    if trend_names is not None:
        for trend_name in trend_names:
            if trend_name == 'bollinger_bands':
                trend_indicators[trend_name] = indicators.Bollinger_Bands(prices, period=20, upperB_p=2, lowerB_p=2)
            if trend_name == 'MACD':
                trend_indicators[trend_name] = indicators.MACD(prices, period=(12,26), ma_p=9)
            if trend_name == 'RSI':
                trend_indicators[trend_name] = indicators.RSI(prices, period=14)
    if status_names is not None:
        for status_name in status_names:
            if status_name == 'RSI':
                status_indicators[status_name] = indicators.RSI(prices, period=14)
    return trend_indicators, status_indicators

def data_regularize(prices, spliter, trend_indicators, status_indicators, percentage):
    assert(isinstance(prices, dict))
    assert (isinstance(trend_indicators, dict))
    assert (isinstance(status_indicators, dict))
    train_set = {}
    test_set = {}
    # get required values from indicators
    for key in list(trend_indicators.keys()):
        trend_indicators[key].cal_data()
        #required_data = trend_indicators[key].getData()
        #prices.update(required_data)
    for key in list(status_indicators.keys()):
        status_indicators[key].cal_data()
        #required_data = status_indicators[key].getData()
        #prices.update(required_data)
    train_set, test_set = spliter.split_data(prices, percentage=percentage)
    # update the cutoff offset for each indicators
    for key in list(trend_indicators.keys()):
        trend_indicators[key].cutoff = spliter.offset
    for key in list(status_indicators.keys()):
        status_indicators[key].cutoff = spliter.offset
    return train_set, test_set

def read_bundle_csv(path, sep=',', filter_data=True, fix_open_price=False, percentage=0.8, extra_indicator=False, trend_names=[], status_names=[]):
    reader = csv_reader()
    spliter = SimpleSpliter()
    train_set = {}
    test_set = {}
    extra_set = {}
    file_list = os.listdir(path)
    for file_name in file_list:
        indicator_dicts = {} # extra_set = {"0005.HK": {"trend", "status"}, "0011.HK": {"trend", "status"}, ...}
        required_path = path + "/" + file_name
        prices = reader.read_csv(required_path, sep=sep, filter_data=filter_data, fix_open_price=fix_open_price)
        if extra_indicator:
            indicator_dicts['trend'], indicator_dicts['status'] = addition_indicators(prices, trend_names, status_names)
            extra_set[file_name] = indicator_dicts
            # data regularize
            train_set_, test_set_ = data_regularize(prices, spliter, indicator_dicts['trend'], indicator_dicts['status'], percentage=percentage)
        else:
            train_set_, test_set_ = spliter.split_data(prices, percentage=percentage)
        train_set[file_name] = train_set_
        test_set[file_name] = test_set_
    print("Totally, read done, got %d rows, %d filtered, %d open prices adjusted" % (
        reader.total_count_filter + reader.total_count_out, reader.total_count_filter, reader.total_count_fixed))
    print("The whole data set size for training: %d and for evaluation: %d" %(
        spliter.trainSet_size, spliter.testSet_size))

    return train_set, test_set, extra_set

class gan_data_container:
    def __init__(self, universe_price_data, universe_extra_set, shapeList_, train_mode, required_volume=False):
        self.universe_price_data = universe_price_data
        self.universe_extra_set = universe_extra_set
        self.price_shape, self.trend_shape, self.status_shape = shapeList_
        self.bars_count = self.price_shape[0]
        self.train_mode = train_mode
        self.required_volume = required_volume
        self.seed()
        self.choose_instrument()
        self.available_start = self.find_available_start()

    def choose_instrument(self):
        self._instrument = self.np_random.choice(list(self.universe_price_data.keys()))
        self.price_data = self.universe_price_data[self._instrument]
        self.extra_set = self.universe_extra_set[self._instrument]
        print("--------- We will use the instrument: %s from universe_train_set.---------" %(self._instrument))

    def find_available_start(self):
        available_start = 0
        if len(self.extra_set) is not 0:
            # append the length, cal the min_length
            invalid_length = []
            if len(self.extra_set['trend']) is not 0:
                for key in list(self.extra_set['trend'].keys()):
                    invalid_length.append(self.extra_set['trend'][key].invalid_len)
            if len(self.extra_set['status']) is not 0:
                for key in list(self.extra_set['status'].keys()):
                    invalid_length.append(self.extra_set['status'][key].invalid_len)
            if self.train_mode:
                available_start = np.max(invalid_length)
            else:
                available_start = 0
        print("--------- The mode is: %s, and the available start is: %d. --------- " %(self.train_mode, available_start))
        return available_start

    def normalised_trend_data(self, offset):
        start = offset - self.bars_count + 1
        end = offset + 1
        # normalise the data from an array
        x = 0
        y = 0
        target_data = np.ndarray(shape=(self.bars_count, self.trend_shape[1]), dtype=np.float64)
        for indicator in self.extra_set['trend'].values():
            y = y + indicator.encoded_size
            target_data[:, x:y] = indicator.normalise(start, end, self.train_mode)
            x = y
            y = x
        return target_data

    def normalised_trend_data_one(self, offset):
        x = 0
        y = 0
        target_data = np.ndarray(shape=(1, self.trend_shape[1]), dtype=np.float64)
        for indicator in self.extra_set['trend'].values():
            y = y + indicator.encoded_size
            target_data[0, x:y] = indicator.normalise_one(offset, self.train_mode)
            x = y
            y = x
        return target_data

    def encode(self, offset):
        X_v_buffer = np.zeros(shape=(self.bars_count, self.price_shape[1]), dtype=np.float32)
        K_v_buffer = np.zeros(shape=(self.bars_count, self.trend_shape[1]), dtype=np.float32)
        x_v_buffer = np.zeros(shape=(1, self.price_shape[1]), dtype=np.float32)
        k_v_buffer = np.zeros(shape=(1, self.trend_shape[1]), dtype=np.float32)

        # stacking the data:
        # X_v
        shift_r = 0
        bese_volume = self.price_data['volume'][offset - self.bars_count + 1]
        for bar_idx in range(-self.bars_count + 1, 1):
            shift_c = 0
            X_v_buffer[shift_r, shift_c] = (self.price_data['high'][offset + bar_idx] - self.price_data['open'][offset + bar_idx]) / \
                                     self.price_data['open'][offset + bar_idx]
            shift_c += 1
            X_v_buffer[shift_r, shift_c] = (self.price_data['low'][offset + bar_idx] - self.price_data['open'][offset + bar_idx]) / \
                                     self.price_data['open'][offset + bar_idx]
            shift_c += 1
            X_v_buffer[shift_r, shift_c] = (self.price_data['close'][offset + bar_idx] - self.price_data['open'][offset + bar_idx]) / \
                                     self.price_data['open'][offset + bar_idx]
            shift_c += 1
            X_v_buffer[shift_r, shift_c] = (self.price_data['close'][(offset - 1) + bar_idx] - self.price_data['open'][offset + bar_idx]) / \
                                     self.price_data['open'][offset + bar_idx]
            shift_c += 1
            if self.required_volume:
                X_v_buffer[shift_r, shift_c] = self.price_data['volume'][offset + bar_idx] / bese_volume
                shift_c += 1
            shift_r += 1

        # K_v
        if len(self.extra_set['trend']) is not 0:
            K_v_buffer = self.normalised_trend_data(offset)

        # x_v
        shift_c = 0
        x_v_buffer[0, shift_c] = (self.price_data['high'][offset + 1] - self.price_data['open'][offset + 1]) / \
                                       self.price_data['open'][offset + 1]
        shift_c += 1
        x_v_buffer[0, shift_c] = (self.price_data['low'][offset + 1] - self.price_data['open'][offset + 1]) / \
                                       self.price_data['open'][offset + 1]
        shift_c += 1
        x_v_buffer[0, shift_c] = (self.price_data['close'][offset + 1] - self.price_data['open'][offset + 1]) / \
                                       self.price_data['open'][offset + 1]
        shift_c += 1
        x_v_buffer[0, shift_c] = (self.price_data['close'][(offset - 1) + 1] - self.price_data['open'][offset + 1]) / \
                                       self.price_data['open'][offset + 1]
        shift_c += 1
        if self.required_volume:
            x_v_buffer[1, shift_c] = self.price_data['volume'][offset + 1] / bese_volume
            shift_c += 1

        # k_v
        if len(self.extra_set['trend']) is not 0:
            k_v_buffer = self.normalised_trend_data_one(offset + 1)

        return X_v_buffer, K_v_buffer, x_v_buffer, k_v_buffer

    def generate_batch(self, batch_size):

        # random choose the number in size of batch (PS: -1 for the expected data)
        offsets = self.np_random.choice(range(self.available_start, self.price_data['high'].shape[0] - self.bars_count * 10 - 1), size=batch_size) + self.bars_count

        # define the empty array
        X_v = np.zeros(shape=(batch_size, self.bars_count, self.price_shape[1]), dtype=np.float32)
        K_v = np.zeros(shape=(batch_size, self.bars_count, self.trend_shape[1]), dtype=np.float32)
        x_v = np.zeros(shape=(batch_size, 1, self.price_shape[1]), dtype=np.float32)
        k_v = np.zeros(shape=(batch_size, 1, self.trend_shape[1]), dtype=np.float32)

        for counter, offset in enumerate(offsets):
            X_v[counter,:,:], K_v[counter,:,:], x_v[counter,:,:], k_v[counter,:,:] = self.encode(offset)
        # self.checking(offsets, X_v, K_v, x_v, k_v, self.train_mode)

        # change to tensors
        X_v = torch.tensor(X_v, dtype=torch.float32)
        K_v = torch.tensor(K_v, dtype=torch.float32)
        x_v = torch.tensor(x_v, dtype=torch.float32)
        k_v = torch.tensor(k_v, dtype=torch.float32)
        return X_v, K_v, x_v, k_v

    def checking(self, offsets, X_v, K_v, x_v, k_v, train_mode):
        for b, offset in enumerate(offsets):
            if train_mode is False:
                offset = offset + self.extra_set['trend']['MACD'].cutoff
            print("----------------------\noffset: %d: %s\n Opening: %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f"
                  "\n Ending: %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f"
                  % (offset, train_mode, X_v[b, 0, 0], X_v[b, 0, 1], X_v[b, 0, 2], X_v[b, 0, 3],
                     K_v[b, 0, 0], K_v[b, 0, 1], K_v[b, 0, 2], K_v[b, 0, 3],
                     X_v[b, -1, 0], X_v[b, -1, 1], X_v[b, -1, 2], X_v[b, -1, 3],
                     K_v[b, -1, 0], K_v[b, -1, 1], K_v[b, -1, 2], K_v[b, -1, 3]
                     ))
            print("t+1 data :\n %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f" % (
                x_v[b, 0, 0], x_v[b, 0, 1], x_v[b, 0, 2], x_v[b, 0, 3],
                k_v[b, 0, 0], k_v[b, 0, 1], k_v[b, 0, 2], k_v[b, 0, 3]
            ))
    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

def D_preprocess(X_v, K_v, x_v, k_v):
    W_t0 = torch.cat((X_v, K_v), axis=2).to(torch.device("cpu"))
    W_t1 = torch.cat((x_v, k_v), axis=2).to(torch.device("cpu"))
    W_t = torch.cat((W_t0, W_t1), axis=1).to(torch.device("cpu"))
    return W_t