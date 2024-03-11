## P4DDLe

### Introduction

We use an altered version of LUCID as the P4ddle/Baseline control-plane. The main difference present in this version is the support for P4.

### Prerequisites

- Ubuntu Linux 22.04
- Git
- Python 3.x
- Pip3

### Step 1: Installing Required Packages

First, let's ensure we have the necessary packages installed:

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv
```

### Step 2: Setting Up Python Virtual Environment

Next, let's create and activate a Python virtual environment:

```bash
python3 -m venv venv
source ./venv/bin/activate
# to exit the virtual environment run the command deactivate
```

### Step 3: Installing Dependencies and Mininet

Now, let's install the required Python dependencies and mininet:

```bash
sudo apt install -y tshark mininet tcpreplay
(venv) pip3 install tensorflow==2.8.4 scikit-learn h5py pyshark protobuf==3.19.6 mininet thrift psutil
```

### Step 4: Installing and Compiling P4 Compiler and Behavioral Model

We'll begin by installing the P4 Compiler from its repository:

```bash
source /etc/lsb-release
echo "deb http://download.opensuse.org/repositories/home:/p4lang/xUbuntu_${DISTRIB_RELEASE}/ /" | sudo tee /etc/apt/sources.list.d/home:p4lang.list
curl -fsSL "https://download.opensuse.org/repositories/home:p4lang/xUbuntu_${DISTRIB_RELEASE}/Release.key" | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/home_p4lang.gpg > /dev/null
sudo apt update
sudo apt install p4lang-p4c
```

Unfortunately the BMv2 package to not contain the python libraries needed to run P4ddle, so we need to manually compile it:

```bash
(venv) git clone https://github.com/p4lang/behavioral-model.git
(venv) cd behavioral-model
(venv) ./autogen.sh 
(venv) ./configure --with-python_prefix=$VIRTUAL_ENV
(venv) make -j3
(venv) sudo make install
```

### Step 5: Preparing the Dataset

Before we train our model, we need to prepare our dataset:

```bash
(venv) python3 control-plane/lucid_dataset_parser.py --dataset_type DOS2019 --dataset_folder ../sample-dataset/ --packets_per_flow 10 --dataset_id DOS2019 --traffic_type all --time_window 4 --p4_compatible
(venv) python3 control-plane/lucid_dataset_parser.py --preprocess_folder ../sample-dataset/
```

### Step 6: Training the Deep Learning Model

Now, to train the model, let's run:

```bash
(venv) python3 control-plane/lucid_cnn.py --train ../sample-dataset/
```

### Step 7: Performing Live Predictions

After training the model, we can perform predictions on both sample data and live traffic:

```bash
(venv) python3 control-plane/lucid_cnn.py --predict sample-dataset/ --model sample-dataset/4t-10n-DOS2019-LUCID-p4.h5
(venv) python3 control-plane/lucid_cnn.py --predict_live sample-dataset/CIC-DDoS-2019-DNS.pcap --model sample-dataset/4t-10n-DOS2019-LUCID-p4.h5 --dataset_type DOS2019
```

### Step 8: Initiating P4 Switch Implementation

Now, to run the LUCID with the P4. We need to compile and start it, in a separate terminal:

```bash
# Open a new terminal
cd data-plane/baseline
p4c --target bmv2 --arch v1model --std p4-16 p4_packet_management.p4

(venv) cd ../runner
(venv) sudo python3 launcher.py --behavioral-exe simple_switch --json ../baseline/p4_packet_management.json --cli simple_switch_CLI
```

### Step 9: Generating Traffic

While the P4 switch is running, let's generate traffic for testing:

```bash
# Open a new terminal
(venv) sudo python3 helpers/traffic_generator.py -f sample-dataset/CIC-DDoS-2019-DNS.pcap -i s1-eth2 -d 600
```

### Step 10: Live Predictions with the P4 Switch

Finally, with the switch running and with the traffic generator active, let's perform live predictions using LUCID:

```bash
# In the first terminal
(venv) python3 control-plane/lucid_cnn.py --predict_live localhost:22222 --model sample-dataset/4t-10n-DOS2019-LUCID-p4.h5 --dataset_type DOS2019 -r 14 -si baseline
```

### Conclusion

After some time, you can close the 3 terminals. All the data generate by LUCID will be saved in the log folder.