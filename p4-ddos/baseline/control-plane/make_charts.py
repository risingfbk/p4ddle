from os import listdir, makedirs, stat
from os.path import isfile, join, exists
import csv
from typing import Counter
import matplotlib.pyplot as plt
import collections
import json
import sys
import statistics


VALUES_ALL= {
            "accuracy":"ACC", 
            "prediction time":"TIME(sec)",
            "processing time":"proc(sec)",
            "transmission bitrate (mbps)": "extr(sec)",
            "transmission time": "extr(sec)", 
            "packets":"PACKETS",
            "samples":"SAMPLES",
            "registers_occupation":"registers_occupation",
            "total_packet_captured":"total_packet_captured"
        }


VALUES= {
            "accuracy":"ACC", 
            "prediction time":"TIME(sec)",
            "processing time":"proc(sec)",
            "transmission bitrate (mbps)": "extr(sec)",
            "transmission time": "extr(sec)", 
            "packets":"PACKETS",
            "samples":"SAMPLES"
        }

def get_specific_files(register_size, sampling_value,predictions,directory) :
    """
        return only files witch data have been calculated using a specific register size and a specific sampling_value
    """
    targets=list()
    for filename in predictions:
        if filename.split("_")[2].strip() == register_size and int(filename.split("_")[4].strip().split(".")[0].strip()) == sampling_value :
            targets.append(join(directory,filename))

    return targets




def generate_packet_per_sample_size(register_size, sampling_value, f, directory_plots_good, directory_plots_bad):
    good=Counter()
    bad=Counter()
    total_good_packets=0
    total_bad_packets=0
    total_good_flows=0
    total_bad_flows=0

    with open(f) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=' ', skipinitialspace=True)
        for row in csv_reader:
            tmp=row["packets_per_sample_sizes"]
            tmp=json.loads(tmp)

            for k in tmp['good']:
                total_good_packets+=int(tmp['good'][k])*int(k)
                total_good_flows+=int(tmp['good'][k])
                good[int(k)]+=int(tmp['good'][k])

            for k in tmp['bad']:
                total_bad_packets+=int(tmp['bad'][k])*int(k)
                total_bad_flows+=int(tmp['bad'][k])
                bad[int(k)]+=int(tmp['bad'][k])


    good = collections.OrderedDict(sorted(good.items()))
    bad = collections.OrderedDict(sorted(bad.items()))
    tmp={"good":good, "bad":bad}

    for k in tmp:
        if k == "good":
            directory = directory_plots_good
            m=total_good_packets/total_good_flows
        else:
            directory = directory_plots_bad
            m=total_bad_packets/total_bad_flows

        plt.bar(range(len(tmp[k])), list(tmp[k].values()), align='center')
        plt.xticks(range(len(tmp[k])), list(tmp[k].keys()) )
        plt.xlabel("# of packets per flow")
        plt.ylabel("# of flows ")
        speed = f.split("_")[1].strip()
        plt.title("width:{} sampling:{} speed:{} \n mean of {}  packets per flow".format(register_size,sampling_value,speed,m))
        

        plt.savefig(directory + "/{}-sampling-{}_speed-{}".format(register_size, sampling_value, speed), format='png')
        plt.close()



def avg(values):
    tmp=0
    for i in values:
        tmp+=i
    return tmp/len(values)


def get_property_for_speed(speed, f, y_val, register_size):
    y = list()
    val=VALUES_ALL[y_val]
    with open(f) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=' ', skipinitialspace=True)
        for row in csv_reader:
            tmp = float(row[val])
            if y_val == "transmission bitrate (mbps)":
               tmp=(((48+16+16+16+32+9+16+16+8+16+16+32+32+8) * (2** register_size)) /tmp)/1024/1024 # space(bit)/time(sec)=(Nbit/Xsec) /1024/1024 --> Mbps
            if row['TPR'] == '00nan' or row['FPR'] == '00nan' or row['TNR'] == '00nan' or row['FNR'] == '00nan':
               continue # ignore weird results
            y.append(tmp)

 
    return speed, avg(y)
            

def generate_chart_for_metric(register_size, sampling_value, files, y_val, sub):

    """
      Generate chart for accuracy, for a specific sampling value. This generate a line joining the points of different speeds
    """

    tuples=list()
    speeds=list()
    y_values=list()

    for f in files:
        speed = int(f.split("_")[1].strip())
        tmp = get_property_for_speed(speed,f, y_val, register_size)
        tuples.append(tmp)

    tuples.sort(key=lambda tup: tup[0]) # sort based on speed, order is important in plot

    for i in tuples:
      speeds.append(i[0])
      y_values.append(i[1])

    sub.plot(speeds, y_values, linestyle='--', marker='o', label = "s= 1/{}".format(sampling_value))

    sub.set_xticks(speeds, minor=False)


    return plt
         

def generate_avg_register_occupation(files, sampling_value, register_size, destination_folder):
    avg_occupation_per_sample=dict()
    for f in files:
        speed = int(f.split("_")[1].strip())
        speed, avg_occupation = get_property_for_speed(speed,f, "registers_occupation", register_size)
        avg_occupation_per_sample[speed]=avg_occupation


    avg_occupation_per_sample = collections.OrderedDict(sorted(avg_occupation_per_sample.items()))   

    plt.bar(range(len(avg_occupation_per_sample)), list(avg_occupation_per_sample.values()), align='center')
    plt.xticks(range(len(avg_occupation_per_sample)), list(avg_occupation_per_sample.keys()) )
    plt.xlabel("speeds ")
    plt.ylabel("# avg occupation ")
    plt.title("width:{} sampling:{}".format(register_size,sampling_value))

        

    plt.savefig(destination_folder + "/register_size{}-sampling-{}".format(register_size, sampling_value), format='png')
    plt.close()



def generate_byte_transmitted_in_time_window(files, sampling_value, register_size,destination_folder):
    avg_occupation_per_sample=dict()
    for f in files:
        speed = int(f.split("_")[1].strip())
        speed, avg_occupation = get_property_for_speed(speed,f, "total_packet_captured", register_size)
        avg_occupation_per_sample[speed]=avg_occupation* (48+16+16+16+32+9+16+16+8+16+16+32+32+8) /8/1024


    avg_occupation_per_sample = collections.OrderedDict(sorted(avg_occupation_per_sample.items()))   

    plt.bar(range(len(avg_occupation_per_sample)), list(avg_occupation_per_sample.values()), align='center')
    plt.xticks(range(len(avg_occupation_per_sample)), list(avg_occupation_per_sample.keys()) )
    plt.xlabel("speeds ")
    plt.ylabel("# kbytes transmitted ")
    plt.title("width:{} sampling:{}".format(register_size,sampling_value))

        

    plt.savefig(destination_folder + "/register_size{}-sampling-{}".format(register_size, sampling_value), format='png')
    plt.close()    

def generate_total_packet_captured_in_time_window(files, sampling_value, register_size,destination_folder):
    avg_occupation_per_sample=dict()
    for f in files:
        speed = int(f.split("_")[1].strip())
        speed, avg_occupation = get_property_for_speed(speed,f, "total_packet_captured", register_size)
        avg_occupation_per_sample[speed]=avg_occupation


    avg_occupation_per_sample = collections.OrderedDict(sorted(avg_occupation_per_sample.items()))   

    plt.bar(range(len(avg_occupation_per_sample)), list(avg_occupation_per_sample.values()), align='center')
    plt.xticks(range(len(avg_occupation_per_sample)), list(avg_occupation_per_sample.keys()) )
    plt.xlabel("speeds ")
    plt.ylabel("# packets captured ")
    plt.title("width:{} sampling:{}".format(register_size,sampling_value))

        

    plt.savefig(destination_folder + "/register_size{}-sampling-{}".format(register_size, sampling_value), format='png')
    plt.close()  


if __name__ == '__main__' :

    if len(sys.argv) <= 1:
        print("\n Error, please use: python make_charts.py <folder/with/logs/>\n")
        sys.exit(-1)

    folder=sys.argv[1] 

    charts_folder = folder.split("/log")[0]+"/charts"

    if not exists(charts_folder):
        makedirs(charts_folder)    

    speeds=set()
    registers_size=set()
    sampling=set()

    predictions = [f for f in listdir(folder) if isfile(join(folder, f)) and f.startswith("predictions-")]

    for filename in predictions:
        registers_size.add(filename.split("_")[2].strip())
        sampling.add(int(filename.split("_")[4].strip().split(".")[0].strip()))
 
   
    for register_size in registers_size:
        print("generate for size:", register_size)

        directory_register_width=charts_folder+"/register_width_{}".format(register_size)       
        if not exists(directory_register_width):
            makedirs(directory_register_width)

        # first charts group   
        for val in VALUES:
            print("  generate {}".format(val))
          
            fig=plt.figure()
            sub = plt.subplot(111)
            plt.xlabel("mbps")
            plt.ylabel(val)
            for sampling_value in sampling:
                print("    sampling is done taking one packet over {} (1/{})".format(sampling_value,sampling_value))
                files = get_specific_files(register_size, sampling_value, predictions, folder)
                plt=generate_chart_for_metric(int(register_size.split("x")[0]),sampling_value,files, val,sub)
                
            box = sub.get_position()
            sub.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            sub.legend(loc='center left', title='sampling', bbox_to_anchor=(1, 0.5))
            sub.set_title(val)
            fig.savefig(directory_register_width + "/{}-{}".format(register_size,val), format='png')
            plt.close()


       # second chart group, packets per sample dimensions
        
        directory_plots=directory_register_width+"/packets_per_sample_size/"
        if not exists(directory_plots):
            makedirs(directory_plots)
    
        directory_plots_good=directory_plots+"/good/"
        if not exists(directory_plots_good):
            makedirs(directory_plots_good)
        
        directory_plots_bad=directory_plots+"/bad/"
        if not exists(directory_plots_bad):
            makedirs(directory_plots_bad)

        for sampling_value in sampling:
            print("    build packerts per sample dimentions sampling is done taking one packet over {} (1/{})".format(sampling_value,sampling_value))
            files = get_specific_files(register_size, sampling_value, predictions, folder)
            for f in files:
                generate_packet_per_sample_size(register_size,sampling_value,f, directory_plots_good, directory_plots_bad)


       # third chart, group avg_register_occupation byte sended in time window
        
        directory_plots=directory_register_width+"/avg_reg_occupation/"
        if not exists(directory_plots):
            makedirs(directory_plots)

        for sampling_value in sampling:
            files = get_specific_files(register_size, sampling_value, predictions, folder)
            generate_avg_register_occupation(files, sampling_value, register_size,directory_plots)

        directory_plots=directory_register_width+"/byte_transmitted_in_time_window/"
        if not exists(directory_plots):
            makedirs(directory_plots)
        
        for sampling_value in sampling:
            files = get_specific_files(register_size, sampling_value, predictions, folder)
            generate_byte_transmitted_in_time_window(files, sampling_value, register_size,directory_plots)


        directory_plots=directory_register_width+"/total_packets_captured/"
        if not exists(directory_plots):
            makedirs(directory_plots)
        
        for sampling_value in sampling:
            files = get_specific_files(register_size, sampling_value, predictions, folder)
            generate_total_packet_captured_in_time_window(files, sampling_value, register_size,directory_plots)


            





