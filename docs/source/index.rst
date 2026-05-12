Welcome to *ccatkidlib*'s Documentation!
========================================

The `CCAT Observatory <https://www.ccatobservatory.org/>`_ will employ tens of thousands of kinetic inductance detectors (|KID|) in its first-generation science instrument **Prime-Cam**. 
The immense number of KIDs are read out using `Xilinx ZCU111 Radio Frequency System on a Chip <https://www.amd.com/en/products/adaptive-socs-and-fpgas/evaluation-boards/zcu111.html>`_ (|RFSoC|) boards that are controlled using the `primecam_readout <https://github.com/TheJabur/primecam_readout>`_ firmware and software. 
*ccatkidlib* aims to aggregate the general, low-level readout functions provided by *primecam_readout* into high-level functions tailored towards calibration and readout of KIDs. 
Alongside data acquisition capabilities, *ccatkidlib* provides a suite of tools for analyzing and visualizing KID data collected using RFSoC boards. 
An in-depth overview of CCAT Observatory's readout hardware and software stack can be found `here <https://arxiv.org/pdf/2510.06491>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   installation/index
   daq/index
   analysis/index
   glossary
   api/modules

