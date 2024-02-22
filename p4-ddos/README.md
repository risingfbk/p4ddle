sudo apt update
sudo apt install -y python3-pip python3-venv

python3 -m venv venv
source ./venv/bin/activate

pip3 install tensorflow==2.8.4 scikit-learn h5py pyshark protobuf==3.19.6

