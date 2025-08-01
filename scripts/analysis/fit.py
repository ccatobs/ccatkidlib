import time

from ccatkidlib.analysis.detector import Detector
#sys.path.append('/home/pcs/fitting')
#import resonator_model_v3

detector = Detector(com_to='1.1', stream_timestamp='1750570253', data_dir='cooldown_june/coldload', analysis_cfg = '/home/pcs/git/ccatkidlib/scripts/analysis/analysis_config.yaml')
print(detector.stream.data)