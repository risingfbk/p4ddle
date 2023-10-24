## P4DDLe - Simple Switch (Baseline Scenario)

### Install control-plane dependencies

We use an altered capped version of LUCID as the P4ddle control-plane. The main difference present in this version is the support for P4. We also removed all the options unrelated to the scenario in P4ddle.

LUCID requires the installation of a number of Python tools and libraries. This can be done by using the ```conda``` software environment (https://docs.conda.io/projects/conda/en/latest/).

We suggest the installation of ```miniconda```, a light version of ```conda```. ```miniconda``` is available for MS Windows, MacOSX and Linux and can be installed by following the guidelines available at https://docs.conda.io/en/latest/miniconda.html#. 

Execute the following command and follow the on-screen instructions:

```
bash Miniconda3-latest-Linux-x86_64.sh
```

Then create a new ```conda``` environment based on Python 3.9:

```
conda create -n python39 python=3.9
```

Activate the new ```python39``` environment:

```
conda activate python39
```

And configure the environment with ```tensorflow``` and a few more packages:

```
(python39)$ pip install tensorflow==2.7.1
(python39)$ pip install scikit-learn h5py pyshark protobuf==3.19.6
```

Pyshark is used in the ```lucid_dataset_parser.py``` script for data pre-processing.

Pyshark is just Python wrapper for tshark, meaning that ```tshark``` must be also installed. 

On an Debian-based OS, use the following command:

```
sudo apt install tshark
```

Please note that the current parser code works with ```tshark``` **version 3.2.3 or lower** or **version 3.6 or higher**. Issues have been reported when using intermediate releases such as 3.4.X.

### Install data-plane dependencies:

Install mininet and the necessary Python libs.

```
(python39)$ sudo apt install mininet tcpreplay
(python39)$ pip3 install mininet thrift psutil
```

You need BMv2 and P4C. You can install P4C as a package, but you must compile BMv2, since by default the Python Libs are not available on the package version. 

To install both, you can use the following guides:

1. [BMv2](https://github.com/p4lang/behavioral-model)
2. [P4C](https://github.com/p4lang/p4c)

BMv2 installs also the Python APIs required by the ```control-plane/p4_utils.py``` at line 2 ([Here](https://gitlab.com/Mendozz/master-thesis-ddos-detection-via-ml-and-programmable-data-planes/-/blob/p4-test/control-plane/p4_util.py#L2)). These APIs are not part of the standard path, so you need to add them manually. In this case, they were installed in the Python3.9 folder. These libs - ``` bm_runtime ``` and ``` bmpy_utils ``` - must be installed to run LUCID on the control-plane.

### Settings

#### Settings in the data plane
It is possible to change the register size inside the file  ```data-plane/p4_packet_management.p4```. Edit these lines:

```
#define PACKETS 131072
#define PACKET_COUNTER_WIDTH 17
```
Note: PACKETS must be 2^(PACKET_COUNTER_WIDTH)

#### Settings in the control plane
The control plane has a file called  ```control-plane/experiment_**.sh```. This file must agree with the values of PACKETS and PACKET_COUNTER_WIDTH of the data plane.

The lines to edit are:
```
export register_bits=17x2 #2 blocks of 17 Bits
export register_width=262144 # 2x2^Y
```

### Execution order
First, run the _data plane_ code, then run the code of _control plane_ folder. Nonetheless, each folder has its README.

### Final notes:
There could be problems of encoding with bmpy_utils.py (Installed by BMv2) at line 37. You have to add the encoding as follow:

```
# m.update(L) 
m.update(L.encode('utf-8'))
```
