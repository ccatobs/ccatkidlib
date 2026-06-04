import pickle
import polars as pl
import ccatkidlib.analysis.utils.pair as pair
import copy
import gc

from typing import Callable
from pathlib import Path
from ccatkidlib.analysis.core.network import Network

def multi_dump(network: Network, pickle_name: str, num_segments: int = 1, transform: Callable[Network, Network] | None = None) -> None:
    '''Segment specified ``Network`` and transform/pickle segments individually. Specifically, multiple ``Network`` objects are created, each with a subset of the
    ``Detectors`` in the .data DataFrame

    Args:
        network: ``Network`` object to segment and run transformations on/pickle
        pickle_name: Name of pickle file 
        num_segments: Number of segments to split ``Network`` object into
        transform: Function to run on each segment. Must return a ``Network``
    '''

    if transform is not None and not callable(transform): raise TypeError("'transform' argument must be a function!") # Ensure that transform is a function
    pickle_name = pickle_name.split('.')[0] # Remove .pickle from pickle name if included

    num_dets = network.data.height
    len_segments = int(num_dets/num_segments)
    network.data = (network.data.sort('timestamp')
                                .with_columns(pl.Series('num', range(num_dets))))

    sub_dfs = network.data.iter_slices(len_segments)
    det_dict = copy.copy(network.det_dict)

    network.data = None
    network.det_dict = None
    
    for i, sub_df in enumerate(sub_dfs):
        sub_network = copy.deepcopy(network)
        sub_network.data = sub_df

        dets = sub_df['detector']
        sub_network.det_dict = {k: det_dict.pop(k) for k in list(det_dict.keys()) if k in dets}
        if transform is not None: sub_network = transform(sub_network)
        
        pickle_path = Path(sub_network.pickle_dir) / f'{pickle_name}_{i}.pickle'
        with open(pickle_path, 'wb') as f:
             pickle.dump(sub_network, f, pickle.HIGHEST_PROTOCOL)

        del sub_network, sub_df
        gc.collect()

def multi_load(com_to: str, pickle_name: str, sess_id: str, data_dir: str = '**', date: str = '**', root_data_dir: str = '/', transform: Callable[Network, Network] | None = None) -> Network:
    '''
    '''
    
    if transform is not None and not callable(transform): raise TypeError("'transform' argument must be a function!")
    
    bid, drid = com_to.split('.')
    sess_dir = pair.get_sess_dir(sess_id, data_dir = data_dir, date = date, root_data_dir = root_data_dir)

    pickle_dir = Path(f'{sess_dir}/pickle/B{bid}D{drid}/network/{sess_id}/')
    
    # Remove '.pickle' and pickle file number from name if included (e.g., convert 'pickle_test_1.pickle' into 'pickle_test')
    name_parts = pickle_name.split('.')[0].split('_')
    try:
        pickle_num = int(name_parts[-1])
        name_parts = name_parts[:-1]
    except ValueError:
        pass
    glob = '_'.join(name_parts)

    # Load network
    network = None
    for file in pickle_dir.glob(f'{glob}*.pickle'):
        with open(file, 'rb') as f:
            sub_network = pickle.load(f)
            if transform is not None: sub_network = transform(sub_network)
            if network is None:
                network = sub_network
            else:
                network.data = pl.concat([network.data, sub_network.data], how='vertical') # Combine DataFrames
                network.det_dict = network.det_dict | sub_network.det_dict # Combine dictionaries with mappings        
    return network