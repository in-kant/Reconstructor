Reconstructor 3.1 is a GUI tool to interactively run tomographic data processing and reconstruction to find the optimum parameters.
Most of the actual work is done in TomoPy (which is a back-end of the project).
Currently only parallel beam projection geometry is supported, and it heavily relies on the DxChange data format (a specific structure of hdf5 file). However, any generic hdf5 file cn be used, as long as it contains 3D datasets.

Reconstructor requires specific libraries version. We recommend creating a designated Conda environment with the following libraries versions:

| package | version  |
|=========|==========|
| PyQt5   | 5.15.9   |
| silx    | 2.1.2    |
| psutil  | 5.9.4    |
| numpy   | 1.26.4   |
| h5py    | 3.12.1   |
| tomopy  | 1.15.0   |
| scipy   | 1.13.1   |
| opencv  | 4.6.0.66 |


Internal MAX IV wiki page:
https://wiki.maxiv.lu.se/wiki/DanMAX:_Reconstructor