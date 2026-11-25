# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`ccatkidlib` is a Python library for data acquisition (DAQ) and analysis of kinetic inductance detectors (KIDs) at the CCAT Observatory using Xilinx ZCU111 RFSoC boards. It wraps the lower-level `primecam_readout` firmware/software stack via an OCS (Observatory Control System) agent.

## Installation

```bash
pip install .          # standard install
pip install -e .       # editable install for development
```

## Building Documentation

```bash
cd docs/
pip install -r requirements.txt
make html
# Output: docs/_build/html/index.html
```

There is no test suite.

## Architecture

The library has two independent top-level subsystems:

### 1. Data Acquisition (`ccatkidlib/rfsoc/`)

The `R` class (`rfsoc_daq.py`) is the main entry point for controlling RFSoC boards. It:
- Reads a **system config** (YAML with Jinja2 templating) composed of two concatenated YAML documents: an *external config* and an *IO config*
- Communicates with `primecam_readout` through an OCS agent (`OCSClient`)
- Supports multiple boards and drones (up to 4 drones per board), with configurable parallelism via `parallel_boards` and `parallel_drones`
- Key DAQ methods: `take_vna_sweep()`, `take_target_sweep()`, `take_timestream()`, `write_config_comb()`
- Saves data as `.npy` files or `.g3` files (the latter requires `rfsoc-streamer`)

The system config path defaults to `ccatkidlib/rfsoc/system_config.yaml` but can be overridden with the `CCATKIDLIB_SYSTEM_CONFIG` environment variable.

A *drone* is identified as `"{board_id}.{drone_id}"` (e.g., `"1.2"`). Boards and drones are addressed via a `com_to` parameter accepted by all DAQ methods.

### 2. Analysis (`ccatkidlib/analysis/`)

A class hierarchy for loading and analyzing RFSoC data files:

```
Data (base class: data.py)
├── Sweep (sweep.py)        — frequency sweep data
│   ├── VNA (vna.py)        — full 512 MHz bandwidth sweeps
│   └── Target (target.py)  — sweeps centered on tones
└── Timestream (timestream.py) — time-ordered detector data

Detector (detector.py)      — combines Target + Timestream + VNA for a set of detectors
Network (network.py)        — a full array: list of Detector objects for one drone
```

**Data model:** All data is stored in Polars DataFrames. Per-tone columns follow the naming convention `{prefix}_{data_name}_{tone_number_zero_padded}` (e.g., `I_0007`, `rot_I_0007`). Transformations (e.g., `phase()`, `mag()`, `IQ_rotate()`) add new prefixed columns rather than overwriting, controlled via the `prefix` argument.

**Configuration:** Analysis behavior is driven by `analysis_config.yaml` (also Jinja2-templated YAML). It defines column naming conventions, file paths, logging levels, and pickle/parquet settings. The `analysis_cfg` dict is attached to every `Data` object.

**Data loading:** `Data.__init__` locates files without loading them. The `data` attribute is a `@cached_property` — the actual Polars DataFrame is loaded lazily on first access. Configuration YAML files associated with a data file are found via `analysis/utils/pair.py`.

**Properties:** Per-tone scalar results (Q factors, resonance frequencies, fit parameters, etc.) are stored in a separate Polars DataFrame (`properties`) accessed via `get_properties()`.

### Supporting Modules

- `ccatkidlib/io.py` — file/directory IO, YAML loading (with Jinja2), SSH connections via Fabric
- `ccatkidlib/log.py` — logging setup, `@header`/`@method_timer`/`@function_timer` decorators
- `ccatkidlib/utils.py` — general Python utilities
- `ccatkidlib/rfsoc/arg_utils.py` — parses and routes `com_to` arguments and drone config lookups for the `R` class
- `ccatkidlib/analysis/utils/pair.py` — locates data files and their associated config files in the directory tree
- `ccatkidlib/analysis/utils/dataframe.py` — Polars expression helpers (`create_enums`, `parse_tones`, etc.)
- `ccatkidlib/analysis/utils/multiprocess.py` — ProcessPoolExecutor helpers for per-tone parallelism
- `ccatkidlib/analysis/fit/fit.py` — resonator model fitting (lmfit-based)
- `ccatkidlib/analysis/routines/` — higher-level analysis routines (tuning, flagging, properties)
- `ccatkidlib/scripts/live_plot/` — live detector monitoring plots

## Key Conventions

- **Drone IDs** always take the form `"board_id.drone_id"` (e.g., `"1.2"`, `"2.4"`).
- **Config files** are YAML with Jinja2 templating; use `io.load_config()` to parse them. System configs contain two `---`-separated YAML documents (ext config + IO config).
- **Column naming:** `{optional_prefix}_{data_name}_{zero_padded_tone}`. The number of zero-padding digits is set by `analysis_cfg["tones"]["max_tones"]`.
- **`com_to` kwargs** in `R` methods accept a string, list, or use `self.drone_list` by default; `arg_utils.get_com_to()` normalizes them.
- **`return_prefix`** in `analysis_cfg["runtime"]`: when `True`, analysis methods return `(DataFrame, prefix_string)` tuples; when `False` they return only the DataFrame.
- **Attenuation setting** must run serially (one drone per board at a time) due to serial communication hardware constraints — `parallel_drones` is forced to 1 in `set_atten`/`get_atten`.

## Docstring and Type Hint Conventions

Sphinx is configured with `sphinx.ext.napoleon` (Google-style parsing) and `autoclass_content = 'both'` (class + `__init__` docstrings both appear in the docs).

### Type Hints

- Use the `|` union operator: `str | int`, `T | None` — never `Union[str, int]` or `Optional[T]`
- Lowercase generics: `list[str]`, `dict[str, Any]`
- Use `TypeAlias` from `typing` for complex callable signatures (see `analysis/utils/dataframe.py`)
- Always annotate public function signatures with argument types and a return type
- Repeat the type in the `Args:` docstring block (redundant but intentional for readability)
- Wrap signatures to one parameter per line when they exceed ~88 characters

### Docstring Format (Google style)

```
One-sentence summary.

[Extended description if needed.]

[RST directives if needed]

Args:
    param (type): Description.
    param (type, optional): Description. Defaults to X.
Returns:
    type: Description.
Raises:
    ExceptionType: When this happens.
Note:
    Free-form caveats.
Examples:
    >>> code_example()
```

The `Returns:` block uses `type: Description.` — Napoleon renders this as a typed return entry. The non-standard `return (type):` variant found in some older functions in `pair.py` should be updated to `type:` when those functions are touched.

Internal/private helpers (`_`-prefixed, single-use, or nested) may use a one-line `""" """` docstring.

### Sphinx-Rendered RST in Docstrings

`docs/source/conf.py` defines these global substitutions via `rst_epilog`, available in every Python docstring:

| Substitution | Renders as |
|---|---|
| `\|RFSoC\|` | glossary link: *RFSoC* |
| `\|KID\|` | glossary link: *KID* |
| `\|drone\|` | glossary link: *drone* |
| `\|tone\|` | glossary link: *tone* |
| `\|I\|` | math: *I* |
| `\|Q\|` | math: *Q* |

Additional RST conventions used in docstrings:

- `*italics*` — library/package names: `*Polars*`, `*SciPy*`, `*ccatkidlib*`
- ` ``monospace`` ` — code, variable, and column names: ` ``data_path`` `, ` ``'vna'`` `
- `**bold**` — emphasis on key terms
- `:math:\`expression\`` — inline LaTeX; use a raw string (`r"""`) when the expression contains backslashes
- `.. important::` — critical init/usage constraints
- `.. seealso::` — links to external documentation
- `.. codeauthor:: Name <email>` — in module-level docstrings only
- No `:class:`, `:func:`, `:meth:` cross-reference roles — use ` ``backticks`` ` for identifiers instead

### Canonical Example

```python
def get_data_file(
    com_to: str,
    timestamp: str | int,
    data_type: str,
    data_dir: str = "**",
    date: str = "**",
    sess_id: str = "**",
    root_data_dir: str = "/",
) -> list[str]:
    """Get a *ccatkidlib* data file based on provided path information.

    Args:
        com_to (str): |drone| that took the data. In form ``'Board.Drone'``
        timestamp (str | int): Timestamp of data file
        data_type (str): Type of data file. Should be one of ``'vna'``, ``'targ'``, ``'timestream'``
        data_dir (str, optional): Directory where data is stored. Defaults to wildcard ``'**'``
        date (str, optional): Date data was taken. Defaults to wildcard ``'**'``
        sess_id (str, optional): *ccatkidlib* session ID. Defaults to wildcard ``'**'``
        root_data_dir (str, optional): Root directory where data is stored. Defaults to ``'/'``
    Returns:
        list[str]: Path of found data file. Returns ``'invalid/path'`` if data file not found.
    """
```

### Analysis Library Notes

#### Analysis class types — `TYPE_CHECKING` guards

Never substitute `Any` for a known analysis class. For files that cannot import
an analysis class at runtime (to avoid circular imports), use
`from __future__ import annotations` together with a `TYPE_CHECKING` guard:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ccatkidlib.analysis.core.detector import Detector

def some_routine(det: Detector, ...) -> pl.DataFrame: ...
```

`from __future__ import annotations` defers annotation evaluation so the
guarded import never runs at import time. The `ProcessPoolExecutor` (`ex`)
parameter is a stdlib type with no circular risk — import it directly:

```python
from concurrent.futures import ProcessPoolExecutor
```

Known circular import constraint: `analysis/routines/properties.py` cannot import
`Detector` even under `TYPE_CHECKING` because `detector.py` imports `properties.py`
at the module level. Use `Target` instead (which is what `targ` actually is), since
`target.py` does not import `properties.py`.

#### Numba-decorated functions — no Python type annotations

`@guvectorize` and module-level `@njit` functions cannot carry Python type
annotations. Numba injects its own type system through the decorator; Python
annotations are silently ignored for gufuncs and can raise `TypeError` for njit.
Leave them unannotated by design. Nested `@njit` helpers inside Python functions
are also left unannotated. Affected files: `analysis/core/vna.py`, `analysis/fit/fit.py`.

#### LaTeX math in docstrings — always use raw strings

Python raises `SyntaxWarning: invalid escape sequence` for `\letter` in a normal
string. Any docstring containing `:math:` with backslash LaTeX (`\arctan`, `\pi`,
`\sqrt`, etc.) **must** use `r"""` not `"""`. The `:math:` raw-string note in
§"Additional RST conventions" above applies to every such docstring without exception.

#### `fit/kid_phase_fit/` — excluded from documentation

`analysis/fit/kid_phase_fit/` contains ported Gao-thesis code with `import *`
and backslash-continued function signatures. Do not add docstrings or annotations
to `fit_single_det.py`, `phase_fit.py`, or `utils.py`.
