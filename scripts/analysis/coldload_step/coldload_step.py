import polars as pl
import argparse

from pathlib import Path

from ccatkidlib.analysis.core.network import Network

class ColdloadStep(Network):
    def __init__(self, 
                 com_to: str, 
                 dets: int = -1,
                 sess_ids: str | list[str] | None  = None,
                 analysis_cfg: str = './analysis_config.yaml',
                 include_targs = False,
                 **kwargs):
        super().__init__(com_to, dets = dets, sess_ids = sess_ids, analysis_cfg = analysis_cfg, include_streams = True, include_targs = include_targs, **kwargs)


def eval_args():
    '''
    Evaluate command line arguments specified when run as a script
    '''

    parser = argparse.ArgumentParser(prog='Coldload Step Analysis', description='''Analyze coldload step data''')
    parser.add_argument('-S', '--sess', type=str, default=None, help='Session ID(s) of coldload step data files')
    parser.add_argument('-C', "--cfg", type = str, default='./analysis_config.yaml', help='Path of a analysis configuration file.')
    return parser.parse_args()

if __name__ == '__main__':
    args = eval_args()