
# Dewan Lab Calcium Imaging Data Processing Toolbox

Collection of modules to assist with the processing and analysis of microendoscope 1P recordings collected in the Dewan Lab at Florida State University.
Special thanks to Roberto Vincis for the frameworks of several functions included in the package.

## Installation

### **Important**: The current version of this package requires the installation of the Inscopix Data Processing Software (IDPS) API prior to its installation. Please see [IDPS](https://inscopix.com/software-analysis-miniscope-imaging/) for more information about their software

#### **Note**: It is recommended to use an environment manager such as [Anaconda](https://www.anaconda.com/download) or its popular extension [Mamba](https://github.com/mamba-org/mamba) when installing this package

1) Clone the repository to a directory of your choosing
2) Navigate to the clone directory
3) Execute `pip install .` within the root

## Usage

`TBD`

## Troubleshooting and Setup

- OASIS will need to be built from source using a new version of numpy so pandas is happy
- If you receive an error regarding libGL reference the following link: [Stack Overflow](https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo)
- Additionally, you will need to add the ISX library path to the LD_LIBRARY_PATH; see the following link for instructions to tie this to the conda library [Stack Overflow](https://stackoverflow.com/questions/46826497/conda-set-ld-library-path-for-env-only)
  - You can also symlink the inscopix library path to the environment lib path {miniforge_path}/envs/{env_name}/lib
- Further, you will need to create a symlink between the libisxpublicapi.so file and the site-packages folder
- On Windows, you will likely need to move the IDPS software from Program Files to another not protected folder to allow the build wheels to be created
  - Additionally, it is essential to use the `-e` flag when installing with pip