# Copyright (c) 2020 @ FBK - Fondazione Bruno Kessler
# Author: Roberto Doriguzzi-Corin
# Project: LUCID: A Practical, Lightweight Deep Learning Solution for DDoS Attack Detection
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Sample commands
# Training: python3 lucid_cnn.py --train ./sample-dataset/  --epochs 100
# Testing: python3 lucid_cnn.py --predict ./sample-dataset/ --model ./sample-dataset/10t-10n-SYN2020-LUCID.h5

import sys
import json
import csv
import tensorflow as tf
import numpy as np
import random as rn
import os
from util_functions import *
import p4_util

# Seed Random Numbers
os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
rn.seed(SEED)
config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1)

from itertools import cycle
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv1D, LSTM, Reshape
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential, save_model, load_model, clone_model
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score,accuracy_score,mean_squared_error, log_loss, confusion_matrix
from sklearn.utils import shuffle
from lucid_dataset_parser import *

import tensorflow.keras.backend as K
tf.random.set_seed(SEED)
K.set_image_data_format('channels_last')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)

MODEL_NAME_LEN = 10
TRAINING_HEADER = "Model              TIME(t) ACC(t)  ERR(t)  ACC(v)  ERR(v)  Parameters\n"
VALIDATION_HEADER = "Model            TIME(sec) ACC    ERR    F1     PPV    TPR    FPR    TNR    FNR    Parameters\n"
PREDICTION_HEADER_LIVE = "Model,TIME(sec),PACKETS,SAMPLES,DDOS%,ACC,ERR,F1,PPV,TP,FP,TN,FN,Data Source\n"
PREDICTION_HEADER_BEG = "Model,round_counter,round_time,TIME(sec),PACKETS,SAMPLES,DDOS%,ACC,ERR,F1,PPV,TP,FP,TN,FN,extr(sec),proc(sec),packets_captured_in_memory,total_packet_captured"
PREDICTION_HEADER_END = ",Data_Source\n"
PREDICTION_HEADER_IGN = "ignored_packets"
PREDICTION_HEADER_BASE = PREDICTION_HEADER_BEG+PREDICTION_HEADER_END
PREDICTION_HEADER_FLOWS = PREDICTION_HEADER_BEG+PREDICTION_HEADER_IGN+PREDICTION_HEADER_END
PREDICTION_HEADER_SHORT = "Model            TIME(sec) PACKETS SAMPLES DDOS% Data Source\n"
# hyperparameters
MAX_CONSECUTIVE_LOSS_INCREASE = 10
LR = [0.1,0.01]
BATCH_SIZE = [1024,2048]
KERNELS = [32,64]

def Conv2DModel(model_name, input_shape,kernels,kernel_rows,kernel_col,pool_height='max', regularization=None,dropout=None):
    K.clear_session()

    model = Sequential(name=model_name)
    if regularization == 'l1' or regularization == "l2":
        regularizer = regularization
    else:
        regularizer = None

    model.add(Conv2D(kernels, (kernel_rows,kernel_col), strides=(1, 1), input_shape=input_shape, kernel_regularizer=regularizer, name='conv0'))
    if dropout != None and type(dropout) == float:
        model.add(Dropout(dropout))
    model.add(Activation('relu'))
    current_shape = model.layers[0].output_shape
    current_rows = current_shape[1]
    current_cols = current_shape[2]
    current_channels = current_shape[3]

    # height of the pooling region
    if pool_height == 'min':
        pool_height = 3
    elif pool_height == 'max':
        pool_height = current_rows
    else:
        pool_height = 3

    pool_size = (min(pool_height, current_rows), min(3, current_cols))
    model.add(MaxPooling2D(pool_size=pool_size, name='mp0'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid', name='fc1'))

    print(model.summary())
    return model

def compileModel(model,lr):
    # optimizer = SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False)
    optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])  # here we specify the loss function

def trainingEpoch(model, batch_size, parameters, X_train,Y_train,X_val,Y_val, output_file):
    tt0 = time.time()
    history = model.fit(x=X_train, y=Y_train, validation_data=(X_val, Y_val), epochs=1, batch_size=batch_size, verbose=2, callbacks=[])  # TODO: verify which callbacks can be useful here (https://keras.io/callbacks/)
    tt1 = time.time()

    accuracy_train = history.history['accuracy'][0]
    loss_train = history.history['loss'][0]
    accuracy_val = history.history['val_accuracy'][0]
    loss_val = history.history['val_loss'][0]

    model_name_string = model.name.ljust(MODEL_NAME_LEN)
    time_string_train = '{:10.5f}'.format(tt1-tt0) + " "

    test_string_train = '{:06.5f}'.format(accuracy_train) + " " + '{:06.5f}'.format(loss_train) + " "

    test_string_val = '{:06.5f}'.format(accuracy_val) +  " " + '{:06.5f}'.format(loss_val) + " "
    test_string_parameters = parameters + "\n"

    output_string = model_name_string + time_string_train + test_string_train + test_string_val+ test_string_parameters
    output_file.write(output_string)
    output_file.flush()

    return loss_val, accuracy_val

def trainCNNModels(model_name, epochs, X_train, Y_train,X_val, Y_val, dataset_folder, time_window, max_flow_len,regularization=None, dropout=None):

    packets = X_train.shape[1]
    features = X_train.shape[2]
    best_f1_score = 0

    stats_file = open(dataset_folder + 'training_history-' + time.strftime("%Y%m%d-%H%M%S") + '.csv', 'a')
    stats_file.write(TRAINING_HEADER)

    if epochs == 0:
        epochs_range = cycle([0]) # infinite epochs
        epochs = 'inf'
    else:
        epochs_range = range(epochs)

    for lr in LR:
        for kernels in KERNELS:
            for kernel_rows in [min(3,packets)]:
                for kernel_columns in [features]:
                    for pool_height in ['min','max']: # min=3, max=number of rows after the convolution
                        for batch_size in BATCH_SIZE:
                            stop_counter = 0
                            min_loss = float('inf')
                            max_acc_val = 0
                            parameters = "lr=" + '{:04.3f}'.format(lr) + ",b=" + '{:04d}'.format(batch_size) + ",n=" + '{:03d}'.format(max_flow_len) + ",t=" + '{:03d}'.format(time_window) + ",k=" + '{:03d}'.format(kernels) + ",h=(" + '{:02d}'.format(kernel_rows) + "," + '{:02d}'.format(kernel_columns) + "),m=" + pool_height
                            model = Conv2DModel(model_name, X_train.shape[1:4], kernels, kernel_rows, kernel_columns,pool_height,regularization,dropout)
                            compileModel(model,lr)
                            best_model = None
                            best_model_loss_val = float('inf')
                            epoch_counter = 0
                            for epoch in epochs_range:
                                print("Epoch: %d/%s" % (epoch_counter + 1, str(epochs)))
                                epoch_counter += 1
                                loss_val, acc_val= trainingEpoch(model, batch_size, parameters, X_train, Y_train, X_val, Y_val, stats_file)

                                if acc_val > max_acc_val:
                                    max_acc_val = acc_val
                                    best_model_loss_val = loss_val
                                    best_model = clone_model(model)
                                    best_model.set_weights(model.get_weights())

                                if loss_val > min_loss:
                                    stop_counter += 1
                                else:
                                    min_loss = loss_val
                                    stop_counter = 0

                                if stop_counter > MAX_CONSECUTIVE_LOSS_INCREASE or max_acc_val == 1 or epoch+1 == epochs: # early stopping management
                                    if best_model is not None:
                                        tp0 = time.time()
                                        Y_pred_val = (best_model.predict(X_val) > 0.5)
                                        tp1 = time.time()
                                        Y_true_val = Y_val.reshape((Y_val.shape[0], 1))
                                        acc_score_val = accuracy_score(Y_true_val, Y_pred_val)
                                        ppv_score_val = precision_score(Y_true_val, Y_pred_val)
                                        f1_score_val = f1_score(Y_true_val, Y_pred_val)
                                        tn, fp, fn, tp = confusion_matrix(Y_true_val, Y_pred_val, labels=[0, 1]).ravel()
                                        tnr_score_val = tn / (tn + fp)
                                        fpr_score_val = fp / (fp + tn)
                                        fnr_score_val = fn / (fn + tp)
                                        tpr_score_val = tp / (tp + fn)

                                        model_name_string = best_model.name.ljust(MODEL_NAME_LEN)
                                        time_string_predict = '{:10.3f}'.format(tp1 - tp0) + " "
                                        test_string_val = '{:05.4f}'.format(acc_score_val) + \
                                                          " " + '{:05.4f}'.format(best_model_loss_val) + " " + '{:05.4f}'.format(f1_score_val) + \
                                                          " " + '{:05.4f}'.format(ppv_score_val) + \
                                                          " " + '{:05.4f}'.format(tpr_score_val) + " " + '{:05.4f}'.format(fpr_score_val) + \
                                                          " " + '{:05.4f}'.format(tnr_score_val) + " " + '{:05.4f}'.format(fnr_score_val) + " "

                                        test_string_parameters = parameters + "\n"

                                        output_string = model_name_string + time_string_predict + test_string_val + test_string_parameters

                                        if f1_score_val > best_f1_score: #save new best model along with its stats and parameters
                                            try:
                                                filename = dataset_folder + str(time_window) + 't-' + str(
                                                    max_flow_len) + 'n-' + best_model.name
                                                best_model.save(filename + '.h5')
                                                model_stats_file = open(filename + '.csv', 'w')
                                                model_stats_file.write(VALIDATION_HEADER)
                                                model_stats_file.write(output_string)
                                                model_stats_file.flush()
                                                model_stats_file.close()
                                                best_f1_score = f1_score_val
                                            except:
                                                print("An exception occurred when saving the model!")
                                        del best_model

                                    del model
                                    break

    stats_file.close()

def main(argv):


    class ActionToPreType(argparse.Action):
        def __init__(self, option_strings, dest, nargs=None, **kwargs):
            if nargs is not None:
                raise ValueError("nargs not allowed")
            super(ActionToPreType, self).__init__(
                option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            def from_str(value):
                packet_replication_values = {'SimplePre':1, 'SimplePreLAG':2, 'None':3 }
                try:
                  return packet_replication_values[value]
                except:
                  return 1 # SimplePre

            assert(type(values) is str)
            setattr(namespace, self.dest, from_str(values))

    help_string = 'Usage: python3 lucid_cnn.py --train <dataset_folder> -e <epocs>'

    parser = argparse.ArgumentParser(
        description='DDoS attacks detection with convolutional neural networks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-t', '--train', nargs='+', type=str,
                        help='Start the training process')

    parser.add_argument('-e', '--epochs', default=0, type=int,
                        help='Training iterations')

    parser.add_argument('-a', '--attack_net', default=None, type=str,
                        help='Subnet of the attacker (used to compute the detection accuracy)')

    parser.add_argument('-v', '--victim_net', default=None, type=str,
                        help='Subnet of the victim (used to compute the detection accuracy)')

    parser.add_argument('-p', '--predict', nargs='?', type=str,
                        help='Perform a prediction on pre-preocessed data')

    parser.add_argument('-pl', '--predict_live', nargs='?', type=str,
                        help='Perform a prediction on live traffic from network interface, pcap file or p4 compatible switch in format <host>:<port>')

    parser.add_argument('--p4_switch_json_file', help='JSON description of P4 program',
                        type=str, required=False)

    parser.add_argument('--p4_switch_pre', help='Packet Replication Engine used by target',
                        type=str, choices=['None', 'SimplePre', 'SimplePreLAG'],
                        default='SimplePre', action=ActionToPreType)

    parser.add_argument('-i', '--iterations', default=1, type=int,
                        help='Predict iterations')

    parser.add_argument('-m', '--model', type=str,
                        help='File containing the model')

    parser.add_argument('-R', '--regularization', nargs='?', type=str, default=None,
                        help='Apply a regularization technique (l1,l2)')

    parser.add_argument('-r', '--register_bits', nargs='?', type=int, default=None,
                        help='Size of the register in bits')
    
    parser.add_argument('-w', '--register_width', nargs='?', type=str, default=None,
                        help='Size of the registry memory')

    parser.add_argument('-s', '--sampling', nargs='?', type=int, default=None,
                        help='Sampling size')

    parser.add_argument('-an', '--attack_name', nargs='?', type=str, default="live_traffic",
                        help='Attack name to the logs')

    parser.add_argument('-si', '--switch_impl', nargs='?', type=str, choices=['baseline', 'p4ddle'], default="baseline",
                        help='Switch implementation (baseline or p4ddle)')

    parser.add_argument('-d', '--dropout', nargs='?', type=float, default=None,
                        help='Apply dropout to the convolutional layer')

    parser.add_argument('-y', '--dataset_type', default=None, type=str,
                        help='Type of the dataset. Available options are: IDS2017, IDS2018, SYN2020')

    args = parser.parse_args()

    if args.train is not None:
        subfolders = glob.glob(args.train[0] +"/*/")
        if len(subfolders) == 0: # for the case in which the is only one folder, and this folder is args.dataset_folder[0]
            subfolders = [args.train[0] + "/"]
        else:
            subfolders = sorted(subfolders)
        for full_path in subfolders:
            full_path = full_path.replace("//", "/")  # remove double slashes when needed
            folder = full_path.split("/")[-2]
            dataset_folder = full_path
            X_train, Y_train = load_dataset(dataset_folder + "/*" + '-train.hdf5')
            X_val, Y_val = load_dataset(dataset_folder + "/*" + '-val.hdf5')

            X_train, Y_train = shuffle(X_train, Y_train, random_state=SEED)
            X_val, Y_val = shuffle(X_val, Y_val, random_state=SEED)

            # get the time_window and the flow_len from the filename
            train_file = glob.glob(dataset_folder + "/*" + '-train.hdf5')[0]
            filename = train_file.split('/')[-1].strip()
            time_window = int(filename.split('-')[0].strip().replace('t', ''))
            max_flow_len = int(filename.split('-')[1].strip().replace('n', ''))
            dataset_name = filename.split('-')[2].strip()
            p4 = "-p4" if filename.split('-')[4].strip() == "p4" else ""

            print ("\nCurrent dataset folder: ", dataset_folder)

            trainCNNModels(dataset_name + "-LUCID" + p4, args.epochs,X_train,Y_train,X_val,Y_val,dataset_folder, time_window, max_flow_len, args.regularization, args.dropout)


    if args.predict is not None:
        if os.path.isdir("./log") == False:
            os.mkdir("./log")
        stats_file = open('./log/predictions-' + time.strftime("%Y%m%d-%H%M%S") + '.csv', 'a')
        stats_file.write(PREDICTION_HEADER_LIVE)

        iterations = 1
        if args.iterations is not None and args.iterations > 0:
            iterations = args.iterations

        dataset_filelist = glob.glob(args.predict + "/*test.hdf5")

        if args.model is not None:
            model_list = [args.model]
        else:
            model_list = glob.glob(args.predict + "/*.h5")

        for model_path in model_list:
            model_filename = model_path.split('/')[-1].strip()
            filename_prefix = model_filename.split('-')[0].strip() + '-' + model_filename.split('-')[1].strip() + '-'
            model_name_string = model_filename.split(filename_prefix)[1].strip().split('.')[0].strip()
            model = load_model(model_path)

            # warming up the model (necessary for the GPU)
            warm_up_file = dataset_filelist[0]
            filename = warm_up_file.split('/')[-1].strip()
            if filename_prefix in filename:
                X, Y = load_dataset(warm_up_file)
                Y_pred = np.squeeze(model.predict(X, batch_size=2048) > 0.5)

            for dataset_file in dataset_filelist:
                filename = dataset_file.split('/')[-1].strip()
                if filename_prefix in filename:
                    X, Y = load_dataset(dataset_file)
                    [packets] = count_packets_in_dataset([X])

                    Y_pred = None
                    Y_true = Y
                    avg_time = 0
                    for iteration in range(iterations):
                        pt0 = time.time()
                        Y_pred = np.squeeze(model.predict(X, batch_size=2048) > 0.5)
                        pt1 = time.time()
                        avg_time += pt1 - pt0

                    avg_time = avg_time / iterations

                    report_results(np.squeeze(Y_true), Y_pred, packets, model_name_string, filename, stats_file, avg_time, short_header=True)

    if args.predict_live is not None:

        if os.path.isdir("./log") == False:
            os.mkdir("./log")

        log_time = time.strftime("%Y%m%d-%H%M%S") 
        stats_filename = './log/predictions-' + log_time
        packet_filename = './log/packets-' + log_time
        if args.attack_name:
            stats_filename += '_' + args.attack_name
            packet_filename += '_' + args.attack_name
        if args.register_bits:
            stats_filename += '_' + str(args.register_bits)
            packet_filename += '_' + str(args.register_bits)
        if args.register_width:
            stats_filename += '_' + args.register_width
            packet_filename += '_' + args.register_width
        if args.sampling:
            stats_filename += '_' + args.sampling
            packet_filename += '_' + args.sampling
        stats_filename += '.csv'
        packet_filename += '.csv'

        stats_file = open(stats_filename, "a")

        if args.predict_live is None:
            print("Please specify a valid network interface, pcap file or p4-compatible switch address!")
            exit(-1)

        elif args.predict_live.endswith('.pcap'):
            pcap_file = args.predict_live
            traffic_source = pyshark.FileCapture(pcap_file)
            data_source = pcap_file.split('/')[-1].strip()
            stats_file.write(PREDICTION_HEADER_LIVE)

        elif args.predict_live.find(':') != -1:
            host,port = args.predict_live.split(':')
            pre = args.p4_switch_pre
            device_config = args.p4_switch_json_file
            traffic_source = p4_util.get_runtime_API(host,port,pre,device_config)
            data_source = args.predict_live
            if args.switch_impl == "p4ddle":
                stats_file.write(PREDICTION_HEADER_FLOWS)
            else:
                stats_file.write(PREDICTION_HEADER_BASE)
            packet_file = open(packet_filename, "a")
            packet_writer = csv.writer(packet_file, delimiter=',')

        else:
            traffic_source = pyshark.LiveCapture(interface=args.predict_live)
            data_source = args.predict_live
            stats_file.write(PREDICTION_HEADER_LIVE)

        print ("Prediction on network traffic from: ", data_source)

        # load the labels, if available
        labels = parse_labels(args.dataset_type, args.attack_net, args.victim_net)

        # do not forget command sudo ./jetson_clocks.sh on the TX2 board before testing
        if args.model is not None and args.model.endswith('.h5'):
            model_path = args.model
        else:
            print ("No valid LUCID model specified!")
            sys.exit(-1)

        model_filename = model_path.split('/')[-1].strip()
        filename_prefix = model_filename.split('n')[0] + 'n-'
        time_window = int(filename_prefix.split('t-')[0])
        max_flow_len = int(filename_prefix.split('t-')[1].split('n-')[0])
        model_name_string = model_filename.split(filename_prefix)[1].strip().split('.')[0].strip()
        model = load_model(args.model)
        p4_compatible = True if model_filename.find('p4') > 0 else False

        mins, maxs = static_min_max(p4_compatible,time_window)

        stoping_time = 0
        last_collect_time = 0

        try:
            while (True):
                if time.time() - last_collect_time >= 2.0:

                    round_time = time.time()
                    
                    if isinstance(traffic_source, p4_util.RuntimeAPI):
                        samples, process_time, trasmission_time, packets_per_sample_sizes, avg_packets_in_registers_in_round, total_packet_captured_in_round, round_time, round_counter =  \
                            process_live_traffic(traffic_source, args.dataset_type, labels, max_flow_len, p4_compatible, traffic_type="all", time_window=time_window, switch_impl=args.switch_impl, register_bits=args.register_bits)
                    elif isinstance(traffic_source, pyshark.FileCapture):
                        samples, last_round = process_live_traffic(traffic_source, args.dataset_type, labels, max_flow_len, p4_compatible, traffic_type="all", time_window=time_window)
                    else:
                        samples = process_live_traffic(traffic_source, args.dataset_type, labels, max_flow_len, p4_compatible, traffic_type="all", time_window=time_window)

                    if len(samples) > 0:
                        X,Y_true,keys = dataset_to_list_of_fragments(samples)
                        X = np.array(normalize_and_padding(X, mins, maxs, max_flow_len))

                        if labels is not None:
                            Y_true = np.array(Y_true)
                        else:
                            Y_true = None

                        X = np.expand_dims(X, axis=3)
                        pt0 = time.time()
                        Y_pred = np.squeeze(model.predict(X, batch_size=2048) > 0.5,axis=1)
                        pt1 = time.time()
                        prediction_time = pt1 - pt0
                        [packets] = count_packets_in_dataset([X])

                        if isinstance(traffic_source, p4_util.RuntimeAPI):
                            report_results(Y_true, Y_pred, packets, model_name_string,
                                        data_source, stats_file, prediction_time,process_time, trasmission_time,packets_per_sample_sizes,
                                        avg_packets_in_registers_in_round, total_packet_captured_in_round, round_time, round_counter, packet_writer, args.switch_impl)
                        else:
                            report_results(Y_true, Y_pred, packets, model_name_string, data_source, stats_file, prediction_time, short_header=True)
                            if isinstance(traffic_source, pyshark.FileCapture):
                                if last_round:
                                    sys.exit(0)
        
                        last_collect_time = round_time

        except KeyboardInterrupt:
            print("Ctrl+C pressed. Exiting gracefully.")
            sys.exit(0)

def report_results(Y_true, Y_pred,packets, model_name, dataset_filename, stats_file,prediction_time,process_time=0, 
    trasmission_time=0,packets_per_sample_sizes=0, avg_packets_in_registers_in_round=0, total_packet_captured_in_round=0, round_time=0, round_counter=0, packet_writer=0, short_header=False, switch_impl=None):
    
    ddos_rate = '{:04.3f}'.format(sum(Y_pred)/Y_pred.shape[0])

    if short_header:
        time_string_predict = '{:05.3f}'.format(prediction_time)
        performance_string = '{:07.0f}'.format(packets) + "," + '{:07.0f}'.format(Y_pred.shape[0]) + "," + ddos_rate
    else:
        time_string_predict = str(round_counter) + "," + '{:10.4f}'.format(round_time) + "," + '{:05.3f}'.format(prediction_time)
        performance_string = '{:07.0f}'.format(packets) + "," + '{:07.0f}'.format(Y_pred.shape[0]) + "," + ddos_rate

    if Y_true is not None: # if we have the labels, we can compute the classification accuracy
        Y_true = Y_true.reshape((Y_true.shape[0], 1))
        accuracy = accuracy_score(Y_true, Y_pred)
        try:
            loss = log_loss(Y_true, Y_pred)
        except:
            loss = 0
        ppv = precision_score(Y_true, Y_pred)
        f1 = f1_score(Y_true, Y_pred)
        tn, fp, fn, tp = confusion_matrix(Y_true, Y_pred,labels=[0,1]).ravel()

        if short_header:
            test_string_pre = '{:05.4f}'.format(accuracy) + \
                            "," + '{:05.4f}'.format(loss) + "," + '{:05.4f}'.format(f1) + \
                            "," + '{:05.4f}'.format(ppv) + \
                            "," + '{:04d}'.format(tp) + "," + '{:04d}'.format(fp) + \
                            "," + '{:04d}'.format(tn) + "," + '{:04d}'.format(fn) + \
                            "," + dataset_filename + "\n"
            output_header = PREDICTION_HEADER_LIVE[:-1]
            output_string = model_name + "," + time_string_predict + "," + performance_string + "," + test_string_pre
        else:
            if switch_impl == "p4ddle":
                test_string_pre = '{:05.4f}'.format(accuracy) + \
                          "," + '{:05.4f}'.format(loss) + "," + '{:05.4f}'.format(f1) + \
                          "," + '{:05.4f}'.format(ppv) + \
                          "," + '{:04d}'.format(tp) + "," + '{:04d}'.format(fp) + \
                          "," + '{:04d}'.format(tn) + "," + '{:04d}'.format(fn) + \
                          "," + '{:05.4f}'.format(trasmission_time) + "," + '{:05.4f}'.format(process_time) + \
                          "," + '{:06d}'.format(avg_packets_in_registers_in_round) + "," + '{:06d}'.format(total_packet_captured_in_round) + "," + '{:06d}'.format(ignored_packets) +\
                          "," + dataset_filename + "\n"
                output_header = PREDICTION_HEADER_FLOWS[:-1]
            else:
                test_string_pre = '{:05.4f}'.format(accuracy) + \
                                "," + '{:05.4f}'.format(loss) + "," + '{:05.4f}'.format(f1) + \
                                "," + '{:05.4f}'.format(ppv) + \
                                "," + '{:04d}'.format(tp) + "," + '{:04d}'.format(fp) + \
                                "," + '{:04d}'.format(tn) + "," + '{:04d}'.format(fn) + \
                                "," + '{:05.4f}'.format(trasmission_time) + "," + '{:05.4f}'.format(process_time) + \
                                "," + '{:06d}'.format(avg_packets_in_registers_in_round) + "," + '{:06d}'.format(total_packet_captured_in_round) +\
                                "," + dataset_filename + "\n"
                output_header = PREDICTION_HEADER_BASE[:-1]
            output_string = model_name + "," + time_string_predict + "," + performance_string + "," + test_string_pre

        if packet_writer:
            packet_writer.writerows(packets_per_sample_sizes)  

        stats_file.write(output_string)
        stats_file.flush()
    else:
        output_header = PREDICTION_HEADER_SHORT[:-1]
        output_string = model_name + time_string_predict + performance_string + dataset_filename + "\n"
    print(output_header)
    print(output_string)

if __name__ == "__main__":

    main(sys.argv[1:])
