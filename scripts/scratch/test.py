import multiprocessing as mp
import concurrent.futures
import time
import polars as pl

import ccatkidlib.analysis.utils.dataframe as ccat_df
from ccatkidlib.analysis.core.detector import Detector


def tone_placements(com_to):
    det = Detector(com_to=com_to,  sess_id = '1760727414', date='20251017', stream_timestamp='1760727829', data_dir='coldload', root_data_dir='/ccat/data-280GHz/md0/cooldown_june')
    
    #det.cable_delay
    det.targ.mag(prefix='', dB=True)
    det.targ.savgol(col_name='mag', prefix='dB', window=9, k=1, deriv=0, max_workers=1, recalc=False)
    #det.IQ_unwind(prefix='', data='targ', delay_col = 'network_cable_delay', recalc=False)
    #det.IQ_trim(prefix='unwind_rotate',  window=2, use_fit=False, mag_prefix='savgol0_dB', recalc=False)
    #det.targ.savgol(col_name='I', prefix='tail_trim_unwind_rotate', window=21, k=1, deriv=1, max_workers=1)
    #det.targ.savgol(col_name='Q', prefix='tail_trim_unwind_rotate', window=21, k=1, deriv=1, max_workers=1)
    #det.targ.mag(prefix='savgol1_tail_trim_unwind_rotate', dB=False)

    #diff_IQ = det.targ.get_data('savgol1_tail_trim_unwind_rotate_mag', strict=True).unpivot(value_name='IQ', variable_name='temp').drop('temp')
    #f = det.targ.get_data(['sample', 'f'], strict=True).unpivot(index='sample', value_name='f', variable_name='det').with_columns((pl.col('det').str.strip_prefix('f_')).cast(pl.Int32))
    #full_df = pl.concat([f, diff_IQ], how='horizontal')
    #max_IQ = full_df.filter(~pl.col('IQ').is_nan()).filter((pl.col('IQ') == pl.col('IQ').max()).over('det')).select([pl.col('det'), pl.col('sample').alias('max_IQ_sample'), pl.col('f').alias('max_IQ_f')])
    #shared_cols = ['max_IQ_f', 'max_IQ_sample'] if 'max_IQ_f' in det.properties.schema else []
    #det._properties_df = ccat_df.coalesce_join(det._properties_df, max_IQ, on='det', shared_cols=shared_cols)
    return None
    return det.get_properties('max_IQ_f').to_numpy().T[1]

if __name__ == '__main__':
    spawn_context = mp.get_context("forkserver")
    
    print('starting')
    start = time.time()

    com_tos = ['1.1', '1.2', '2.1', '2.3', '3.1', '3.3']
    #results_dict = {com_to: tone_placements(com_to) for com_to in com_tos}
    ps = [spawn_context.Process(target=tone_placements, args=(com_to,)) for com_to in com_tos]

    for p in ps: p.start()
    for p in ps: 
        p.join() 
        print('Done')

    # with concurrent.futures.ProcessPoolExecutor(max_workers=3, mp_context=spawn_context) as executor:
    #     future_to_batch = {executor.submit(tone_placements, com_to): com_to for com_to in com_tos}
    #     for future in concurrent.futures.as_completed(future_to_batch):
    #         com_to = future_to_batch[future]
    #         tone_freqs = future.result()
    #         results_dict[com_to] = tone_freqs
    
    print(time.time() - start)