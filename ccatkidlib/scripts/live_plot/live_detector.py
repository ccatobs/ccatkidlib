import polars as pl

from ccatkidlib.analysis.core.timestream import Timestream
from ccatkidlib.analysis.core.detector import Detector

class LiveTimestream(Timestream):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = pl.DataFrame({})

class LiveDetector(Detector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)