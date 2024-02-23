import sys
import argparse
import subprocess
import re
import os
import time
import psutil

def main():
    args=parse_input()
    replay_pcap(args.pcap_file, args.interface, args.attack_name, args.benign_file, args.attack_duration, args.attack_packets)


def replay_pcap(pcap_file, interface, attack_name, benign_file, attack_duration, attack_packets):

    tend=time.time()+60*int(attack_duration)
    print("start replaing traffic for {} minutes".format(attack_duration))
    if not os.path.exists(pcap_file):
        print("Pcap file doesn't exist")
        sys.exit(-1)
  
    print("start to play the trace %s - %s"%(attack_name, attack_packets))
    time.sleep(10)
    replay=subprocess.Popen(['/usr/bin/tcpreplay', '-K', '--intf1', interface, '--loop', '100', pcap_file], stdout=subprocess.DEVNULL)
    replay = psutil.Process(replay.pid)
    if attack_packets != "0":
        benign = subprocess.Popen(['/usr/bin/tcpreplay', '-K', '--intf1', interface, '--loop', '100','--pps', attack_packets, benign_file], stdout=subprocess.DEVNULL)
        benign = psutil.Process(benign.pid)

    while time.time() <= tend:
        time.sleep(1)
        if replay.status() == psutil.STATUS_ZOMBIE or replay.status() == psutil.STATUS_DEAD:
            replay=subprocess.Popen(['/usr/bin/tcpreplay', '-K', '--intf1', interface, pcap_file], stdout=subprocess.DEVNULL)
            replay = psutil.Process(replay.pid)

    replay.kill()
    if attack_packets != "0":
        benign.kill()

    print("trace finished!")


def parse_input():
    class ParseEthernetAddress(argparse.Action):
        def __init__(self, option_strings, dest, nargs=None, **kwargs):
            if nargs is not None:
                raise ValueError("nargs not allowed")
            super().__init__(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, value, option_string=None):
            if re.match("[0-9a-f]{2}([:]?)[0-9a-f]{2}(\\1[0-9a-f]{2}){4}$", value.lower()):
                setattr(namespace, self.dest, value)
            else:
                raise argparse.ArgumentError(self, "Wrong Address format")


    parser = argparse.ArgumentParser(description='Replay pcap files for a specific period of time')

    parser.add_argument('-f','--pcap_file', type=str, help='Pcap file to replay')
    parser.add_argument('-i','--interface', type=str, help='Interface used to send traffic')
    parser.add_argument('-a','--attack_name', type=str, help='Name of the attack')
    parser.add_argument('-b','--benign_file', type=str, help='Benign trace')
    parser.add_argument('-d','--duration', type=str, help='Attack duration')
    parser.add_argument('-p','--attack_packets', type=str, help='Num packets')
   
    return parser.parse_args()



if __name__ ==  "__main__":
    main()

