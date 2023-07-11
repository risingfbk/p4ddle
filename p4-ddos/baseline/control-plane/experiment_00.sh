#!/bin/bash

start=$(date)
echo "start $start" >> times.txt

export folder=../../dataset/TestSet-00/

declare -a attack_names=(dns mssql snmp ldap netbios portmap ssdp udplag)
declare -a attack_traces=(dns-chunk2.pcap mssql-chunk2.pcap snmp-chunk1.pcap ldap-chunk2.pcap netbios-chunk2.pcap portmap-chunk-full.pcap ssdp-chunk2.pcap udplag-chunk5.pcap)
declare -a benign_packets=(0 0 0 0 0 0 0 0)

declare -a attack_speed=6

export benign_trace=../../dataset/Benign/dataset-benign0.pcap

export target_interface=s1-eth2 # bmv2 interface

export register_bits=14x2 #bits 17 bits for 2 blocks
export register_width=32768 # 2x2^bits eg. 2*2^17

declare -a sampling_values=(1)
export dataset_type=DOS2019

model=../../dataset/10t-4n-DOS2019-cnn.h5

for i in $(seq 0 $((${#attack_names[*]}-1)))
do
	for sampling in "${sampling_values[@]}"
	do
		export attack_name=${attack_names[$i]}
		export pcap_file=$folder${attack_traces[$i]}
		export speed=${attack_speed}
		export benign_trace
		export sampling
		export attack_packets=${benign_packets[$i]}
		echo "start lucid with the trace $attack_name, with sampling at 1 packet over $sampling"
		echo "register_write sampling_treshold 0 $sampling" | simple_switch_CLI --thrift-port 22222 # set sampling rate in switch
		python3 lucid_cnn.py --predict_live localhost:22222 --model $model --dataset_type $dataset_type
		echo ""
		echo ""
		echo ""
		echo ""
		echo ""
		echo ""

		echo "reset registers in switch"
		echo "register_read num_packets 0" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset num_packets" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset round_counter" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset block_of_registers" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset sampling_counter" | simple_switch_CLI --thrift-port 22222
		sleep 1

		echo "register_reset time0" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset packet_length0" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset tcp_flags0" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset udp_len0" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset dst_ip0" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset ip_flags0" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset src_ip0" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset tcp_len0" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset dst_port0" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset ip_upper_protocol0" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset src_port0" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset tcp_window_size0" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset icmp_type0" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset tcp_ack0" | simple_switch_CLI --thrift-port 22222
		sleep 1

		echo "register_reset time1" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset packet_length1" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset tcp_flags1" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset udp_len1" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset dst_ip1" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset ip_flags1" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset src_ip1" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset tcp_len1" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset dst_port1" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset ip_upper_protocol1" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset src_port1" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset tcp_window_size1" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset icmp_type1" | simple_switch_CLI --thrift-port 22222
		sleep 1
		echo "register_reset tcp_ack1" | simple_switch_CLI --thrift-port 22222
		sleep 1

		echo "reset done"
	done
done

cat /tmp/p4s.s1.log | grep "[I]" | grep log_flow: | cut -d" " -f7 > log/flows_in_runs_${register_bits}_${register_width}.csv

echo "end: $(date)"
echo "end: $(date)" >>  times.txt
