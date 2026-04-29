# ccatkidlib
[Xilinx ZCU111 Radio Frequency System on a Chip](https://www.amd.com/en/products/adaptive-socs-and-fpgas/evaluation-boards/zcu111.html) (RFSoC) based data acquisition and analysis library for the [CCAT Observatory's]((https://www.ccatobservatory.org/) [Prime-Cam](https://www.ccatobservatory.org/prime-cam/) instrument kinetic inductance detectors (KIDs).

## Overview
The CCAT Observatory will employ tens of thousands of kinetic inductance detectors (KIDs) in its first-generation science instrument **Prime-Cam**. The immense number of KIDs are read out using Xilinx ZCU111 Radio Frequency System on a Chip (RFSoC) boards that are controlled using the [*primecam_readout*](https://github.com/TheJabur/primecam_readout) firmware and software. *ccatkidlib* aims to aggregate the general, low-level readout functions provided by *primecam_readout* into high-level functions tailored towards tuning and readout of KIDs. Alongside data acquisition capabilities, *ccatkidlib* provides a suite of tools for analyzing and visualizing KID data collected using RFSoCs. An in-depth overview of CCAT Observatory's readout hardware and software stack can be found [here](https://arxiv.org/pdf/2510.06491).

## Dependencies
*ccatkidlib* depends on the following libraries for data acquisition; no additional libraries are required for analysis:
- **REQUIRED** -- [*primecam_readout*](https://github.com/TheJabur/primecam_readout) - Underlying firmware and software for RFSoC control
- **REQUIRED** -- [*ocs*](https://github.com/simonsobs/ocs?tab=readme-ov-file) - Communication between *ccatkidlib* and *primecam_readout* is mediated via a Observatory Control System (OCS) Agent
- **RECOMMENDED** -- [*rfsoc-streamer*](https://github.com/ccatobs/rfsoc-streamer) - Scalable software used to capture UDP data packets produced by RFSoCs. While *ccatkidlib* has native functionality for capturing these data packets, it does not scale to large numbers of RFSoCs.

## Installation
*ccatkidlib* can be installed from source using pip:
``
git clone https://github.com/ccatobs/ccatkidlib.git
cd ccatkidlib/
pip install .
``
> [!TIP]
Those interested in contributing may wish to install using the -e flag allowing the package to automatically reflect source code updates.

## Documentation
After installation, the full documentation for *ccatkidlib* can be locally built as HTML files using [Sphinx](https://www.sphinx-doc.org/en/master/):
``
cd docs/
make html
``
The root page of the documentation can then be found at `docs/_build/html/index.html`.
