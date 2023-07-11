# General informations


This code belongs to the test branch, it differs from the other code because a few changes have been done in order to perform a set of tests in a simpler manner. 

## Setup

In file _test.sh_ a few parameters have to be set:

* the pcap file of the traffic
* the size of the register, that have to be set accordingly with the contol plane 
* the speeds
* the sampling rates (up to 1 on 16) if is required more you have to change the code in the p4 files.
* the test duration per each combination of speed, sampling and the size of the registers
* the name of the trained model

In order to execute this operation with the help of the script, tcpreplay has to run without root permissions:
```
sudo chmod a+s /usr/bin/tcpreplay
```

This branch requires two additional packages. Activate the new ```python38``` environment:

```
conda activate python38
```

And install new packages: psutil is required in order to automate the tests; thrift is required because we are now working in a conda environment

```
(python38)$ pip3 install psutil thrift
```


#### Include extra APIs
The file ```p4_util.py``` at line 2 includes the extra APIs of the BMv2. They are installed in a folder that depends on your python version: check your python version and fix the path otherwise the script will crash.



## Run
To run the tests just run the script:
```
(python38)$ ./test.sh
```


