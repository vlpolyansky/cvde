# Compactified Voronoi Density Estimation

![CVDE logo](cvde.png)

## Initial setup

### CVDE
The core part of the method is written in C++ with a Python interface. Below is the instruction to compile the code within a conda environment that will later be used for the experiments.

`environment.yml` contains requirements both for the main code and for the provided experiments.

```shell
# Create a conda environment following the requirements
conda env create -f environment.yml
conda activate cvde

# Compile the VDE sources
mkdir build && cd build
cmake CMAKE_TOOLCHAIN_FILE=conda-toolchain.cmake ..
make

# Copy the library to the experiments folder
cp vde.so ../experiments/

# Switch to experiments directory
cd ../experiments/
```

### AWKDE
In order to run comparison experiments with adaptive KDE, you need to install the local version of **awkde**. The original version is available at https://github.com/mennthor/awkde. For fair comparison to keep the same family of kernel functions, the local version does not include data preprocessing such as data normalization.

To install the awkde, please run the following **from the _experiments_ folder** and within the conda environment set up above:
```shell
pip install pybind11
pip install -e ./awkde 
```



## Experiments 
You can run the following commands from the _experiments_ folder:
 - `python vde_vs_cvde.py` <br>
   Perform the comparison of VDE and CVDE on a single gaussian. 
 - `python integral_convergence.py [gaussian|mnist]` <br>
   Perform the experiment that determines the relative stabilization of computed integrals for a provided dataset. 
 - `bash kde_comparison.sh`<br>
   Perform KDE comparison experiments on three datasets: gaussians, anuran calls and MNIST.
 
The plots for all experiments will be generated in the _experiments/images_ folder. 