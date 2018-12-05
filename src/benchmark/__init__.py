import typing as T
import time
import numpy as np


class Measure:
    def __init__(self, k, times_dict: T.Dict[str, T.List[float]]):
        self.k = k
        self.times_dict = times_dict

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        t = time.time()
        self.times_dict[self.k] = self.times_dict.get(self.k, []) + [t - self.start]

    @staticmethod
    def print_times(times: T.Dict[str, T.List[float]]):
        avg = dict(map(lambda k: (k, sum(times[k]) / float(len(times[k]))), times.keys()))
        sums = dict(map(lambda k: (k, sum(times[k])), times.keys()))
        total_avg = sum(avg.values())
        total_sum = sum(sums.values())
        for k in times.keys():
            print("{0}\t{1:.4f}s\t{2:.1f}%\t{3:.1f}%".format(k, avg[k], 100 * avg[k] / total_avg, 100 * sums[k] / total_sum))
        print('\t total {0:.4f}s {1:.4f}'.format(total_avg, total_sum))
        return total_avg, total_sum

def compare(f1, f2, runs):
    times = {}
    for _ in range(runs):
        with Measure('f1', times):
            f1()
        with Measure('f2', times):
            f2()

    return np.mean(times['f1']), np.mean(times['f2'])