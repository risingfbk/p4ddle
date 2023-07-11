This is a test branch. 

This branch has been used to test with the help of scripts a lot of combinations of registers speed and sampling rate. 

There are two folders: _control plane_ contains the files of the controller; _data plane_ contains the p4 code.


### Notes:
You need to install mininet with ``` sudo apt install mininet ``` and also python mininet API and thrift API with ```pip3 install mininet thrift```

You need BMv2 and P4C. You can follow the guides:

1. [BMv2](https://github.com/p4lang/behavioral-model)
2. [P4C](https://github.com/p4lang/p4c)

BMv2 installs also the Python APIs required by the ```control-plane/p4_utils.py``` at line 2 ([Here](https://gitlab.com/Mendozz/master-thesis-ddos-detection-via-ml-and-programmable-data-planes/-/blob/p4-test/control-plane/p4_util.py#L2)). These APIs are not part of the standard path, so you need to add them manually. Mine were installed in Python3.8 folder because i used that Python version.

### Settings

#### Settings in the data plane
It is possible to change the register size inside the file  ```data-plane/p4_packet_management.p4```. Edit these lines:

```
#define PACKETS 131072
#define PACKET_COUNTER_WIDTH 17
```
Note: PACKETS must be 2^(PACKET_COUNTER_WIDTH)

#### Settings in the control plane
The control plane has a file called  ```control-plane/test.sh```. This file must agree with the values of PACKETS and PACKET_COUNTER_WIDTH of the data plane.

The lines to edit are:
```
export register_bits=17x2 #2 blocks of 17 Bits
export register_width=262144 # 2x2^Y
```

### Execution order
First, run the _data plane_ code, then run the code of _control plane_ folder. each folder has its readme.


### Final notes:
There could be problems of encoding with bmpy_utils.py (Installed by BMv2) at line 37. You have to add the encoding as follow:

```
# m.update(L) 
m.update(L.encode('utf-8'))
```
