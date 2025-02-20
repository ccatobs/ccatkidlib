from Resonator import Resonator
import yaml
from pathlib import Path

def get_id(s):
    ret = ''
    flag = 0
    for i in s:
        if i in str([1,2,3,4,5,6,7,8,9,0]):
            ret += i
            flag = 1
        elif flag == 1:
            return ret
    return ret

def get_timestreams(data_path, date, stamp, board):
    '''
    Returns a list of Paths to timestreams and a list
    of Paths to the corresponding cfgs sorted from oldest to newest
    Parameters:
        data_path (Path) : Path object to folder where data all rfsoc data is stored
        date (str) : date in YYYMMDD format
        stamp (str) : timestamp of the data collection session
        board (str) : board and drone number to get data from
    Returns:
        timestreams (list) : an array of Timestream classes
        cfgs (list) : an array of dictionary configs
    '''
    

    timestreams = sorted((data_path/'timestream'/date/board/stamp).glob('*.npy'), key=lambda a: get_id(a.name))
    cfgs = sorted((data_path/'rfsoc'/date/board/stamp).glob('*_stream_config.yaml'), key=lambda a: get_id(a.name))

    return timestreams, cfgs


def get_target_sweeps(data_path, date, stamp, board):
    '''
    Returns a list of Paths to sweeps and a list
    of Paths to the corresponding cfgs sorted from oldest to newest
    Parameters:
        data_path (Path) : Path object to folder where data all rfsoc data is stored
        date (str) : date in YYYMMDD format
        stamp (str) : timestamp of the data collection session
        board (str) : board and drone number to get data from
    Returns:
        timestreams (list) : an array of Sweep classes
        cfgs (list) : an array of dictionary configs
    '''
    

    targs = sorted((data_path/'targ'/date/board/stamp).glob('*.npy'), key=lambda a: get_id(a.name))
    cfgs = sorted((data_path/'rfsoc'/date/board/stamp).glob('*_targ_config.yaml'), key=lambda a: get_id(a.name))

    return targs, cfgs

def load_folder_data(data_path, date, stamp, board):
    '''
    Returns a list of Resonator classes that have all the data in the data session loaded
    in them but not yet processed
    Parameters:
        data_path (Path) : Path object to folder where data all rfsoc data is stored
        date (str) : date in YYYMMDD format
        stamp (str) : timestamp of the data collection session
        board (str) : board and drone number to get data from
    Returns:
        resonators (list) : an array of Resonator classes
    '''
    targs, targ_cfgs = get_target_sweeps(data_path, date, stamp, board)
    timestreams, ts_cfgs = get_timestreams(data_path, date, stamp, board)

    # Open the first target sweep to see how many resonators there are
    with open(targ_cfgs[0]) as config:
        cfg = yaml.safe_load(config)

    resonators = [Resonator(i) for i in range(cfg['rfsoc_tones']['num_tones'])]

    #t1 = time.time()
    for i in range(len(resonators)):
        resonators[i].add_sweep(targs[-1], i, targ_cfgs[-1])

    for f, cfg in zip(timestreams, ts_cfgs):
        for i in range(len(resonators)):
            resonators[i].add_timestream(f, i,  cfg)

    #print(time.time() - t1)

    for res in resonators:
        res.process_timestreams()
    
    return resonators

