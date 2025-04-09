from ctypes import cdll
from pathlib import Path

lib_path = Path(__file__).parent.joinpath('sliding_prob_numba.cpython-313-x86_64-linux-gnu.so')
_sliding_prob = cdll.LoadLibrary(str(lib_path))

def sliding_prob(data, start_range: int, end_range: int):
    print('Running sliding probability via numba!')
    return _sliding_prob.sliding_prob(data, start_range, end_range)