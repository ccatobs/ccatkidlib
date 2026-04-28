import spt3g.core as core
import polars as pl
import numpy as np

def read_timestream(ip_address: str, port: str | int, detector):
    print('running')
    pipe = core.G3Pipeline()
    pipe.Add(core.G3Reader, filename=f"tcp://{ip_address}:{port}")
    pipe.Add(load_frame, detector=detector)
    pipe.Run()
    return True

def load_frame(frame, detector=None, q = None):
    print(frame)
    if frame.type == core.G3FrameType.Scan and detector is not None:
        timestream = detector.stream
        ts, Is, Qs = timestream.load_frame(
            frame, start_time=-1, time_precision=1e8
        )

        ts, Is, Qs = (
            np.array(ts),
            np.array(Is, dtype=np.float64),
            np.array(Qs, dtype=np.float64),
        )

        data = {"sample": range(len(ts))}
        data["t"], data["dt"] = ts, np.array(ts) * 1e9
        for t, I, Q in zip(timestream.tones, Is, Qs):
            data[f"I_{t:0{timestream.padding}d}"] = I
            data[f"Q_{t:0{timestream.padding}d}"] = Q
        timestream._data = pl.DataFrame(data)
        timestream._data = timestream._data.with_columns(
            pl.col("dt").cast(pl.Datetime("ns")),
            (pl.col("t") - pl.col("t").first()).alias("zt"),
        )
        q.put(timestream.data)
