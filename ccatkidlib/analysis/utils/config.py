from contextlib import contextmanager
import ccatkidlib.utils as ccat_utils

@contextmanager
def return_prefix(analysis_cfg):
    ret_prefix = analysis_cfg['runtime']['return_prefix']
    if ret_prefix:
        yield ret_prefix
    else:
        analysis_cfg['runtime']['return_prefix'] = True
        try:
            yield ret_prefix
        finally:
            analysis_cfg['runtime']['return_prefix'] = ret_prefix