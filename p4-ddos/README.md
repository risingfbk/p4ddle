sudo apt update
sudo apt install -y python3-pip python3-venv

python3 -m venv venv
source ./venv/bin/activate

pip3 install tensorflow==2.8.4 scikit-learn h5py pyshark protobuf==3.19.6

we can install the p4 compiler by the official repo, however the bmv2 need to be compile since the python libs are not available on pip or through the debian package, to that:

(venv) ~/$ git clone https://github.com/p4lang/behavioral-model.git

(venv) ~/$ cd behavioral-model
(venv) ~/model/$ ./autogen.sh 
(venv) ~/model/$ ./configure --with-python_prefix=$VIRTUAL_ENV
(venv) ~/model/$ make -j3
(venv) ~/model/$ sudo make install


python3 control-plane/lucid_dataset_parser.py --dataset_type DOS2019 --dataset_folder ../sample-dataset/ --packets_per_flow 10 --dataset_id DOS2019 --traffic_type all --time_window 4 --p4_compatible
python3 control-plane/lucid_dataset_parser.py --preprocess_folder ../sample-dataset/
python3 control-plane/lucid_cnn.py --train ../sample-dataset/