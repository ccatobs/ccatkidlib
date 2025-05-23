from multiprocessing import shared_memory
import numpy as np
import pickle

def init_worker(lock_):
    global lock
    lock = lock_ 

def clear_shared_mem(name):
    try:
        shared_mem = shared_memory.SharedMemory(name=name, create=False)
        shared_mem.close()
        shared_mem.unlink()
        return True
    except FileNotFoundError:
        return False

def frame_worker(i, self, shape, frames_info, masks_info, I_name, Q_name, start_time, time_precision):
    try:
        frames_name, frames_len = frames_info
        masks_name, masks_len = masks_info

        frames_mem, masks_mem = shared_memory.SharedMemory(name=frames_name), shared_memory.SharedMemory(name=masks_name)
        
        frames, masks = pickle.loads(bytes(frames_mem.buf[:frames_len])), pickle.loads(bytes(masks_mem.buf[:masks_len]))

        mask = masks[i]

        t, I, Q = self.load_frame(self, frames[i], start_time, time_precision, mask = mask)

        sums = [np.sum(mask) for mask in masks[:i+1]]
        frame_ind = int(np.sum(sums[:-1]))
        num_samps = int(sums[-1])

        global lock
        with lock:
            I_mem = shared_memory.SharedMemory(name=I_name)
            Q_mem = shared_memory.SharedMemory(name=Q_name)
            Is = np.ndarray(shape, dtype=np.int32, buffer=I_mem.buf)
            Qs = np.ndarray(shape, dtype=np.int32, buffer=Q_mem.buf)

            Is[:, frame_ind:num_samps + frame_ind] = I
            Qs[:, frame_ind:num_samps + frame_ind] = Q

            I_mem.close()
            Q_mem.close()
            frames_mem.close()
            masks_mem.close()
    except Exception as e:
        print(e)
    return i