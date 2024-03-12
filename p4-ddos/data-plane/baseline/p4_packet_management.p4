/* -*- P4_16 -*- */
#include <core.p4>
#include <v1model.p4>

#define PACKETS 16384
#define PACKET_COUNTER_WIDTH 14

//up to 1 packet over 16
#define SAMPLING_COUNTER_WIDTH 4

typedef bit<9>  egressSpec_t;

#include "includes/headers.p4"

#include "includes/registers.p4"

#include "includes/parser.p4"

control MyVerifyChecksum(inout headers hdr, inout metadata meta) {   
    apply {  }
}

control MyIngress(inout headers hdr,
                  inout metadata meta,
                  inout standard_metadata_t standard_metadata) {

    action drop() {
        mark_to_drop(standard_metadata);
    }
    
    action set_output_port(egressSpec_t port) {
        standard_metadata.egress_spec = port;
    }
    
    table change_switch_output_port{ // set switch egress port like a learning bridge algorithm (more bridge less learning)
        key = {
           hdr.ethernet.dstAddr: exact;
        }
        actions = {
            set_output_port;
            drop;
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }
    
    apply {
        
        bit<1> block_new_features;
        bit<1> block_of_registers_value;
        bit<PACKET_COUNTER_WIDTH> packet_counter_value;
        bit<SAMPLING_COUNTER_WIDTH> sampling_counter_value;
        bit<SAMPLING_COUNTER_WIDTH> sampling_treshold_value;
        bit<64> tmp2;
        num_packets.read(tmp2,0);
        if (tmp2 == 0){
            log_msg("log_flow: 0,0,0,0,0,0");
        }
        num_packets.write(0,tmp2+1);

        if (hdr.ipv4.isValid()) {
            
            change_switch_output_port.apply(); // apply "learning bridge"         

        sampling_treshold.read(sampling_treshold_value,0);
        sampling_counter.read(sampling_counter_value,0);
        sampling_counter_value=sampling_counter_value+1;
        sampling_counter.write(0, sampling_counter_value);

        if(sampling_counter_value == sampling_treshold_value) {

                sampling_counter.write(0, 0); // Reset sampling counter
                block_of_registers.read(block_of_registers_value,0);
                if (block_of_registers_value == 0) {

                    bit<64> rc;
                    round_counter.read(rc,0);
                    log_msg("log_flow: {},{},{},{},{},{}",{meta.srcAddr,meta.srcPort, meta.dstAddr, meta.dstPort, meta.ipProtocol, rc});

                    packet_counter0.read(packet_counter_value,0);
                    packet_counter0.write(0, packet_counter_value+1);
                    bit<64> pim;
                    process_in_memory0.read(pim,0);
                    process_in_memory0.write(0,pim+1);
                    time0.write((bit<32>)packet_counter_value, meta.ingress_timestamp); //feature
                    packet_length0.write((bit<32>)packet_counter_value, meta.packet_length); //feature
                    ip_flags0.write((bit<32>) packet_counter_value, meta.ip_flags); //feature
                    tcp_len0.write((bit<32>) packet_counter_value, meta.tcp_len); //feature
                    tcp_ack0.write((bit<32>) packet_counter_value, meta.tcp_ack); //feature
                    tcp_flags0.write((bit<32>) packet_counter_value, meta.tcp_flags); //feature
                    tcp_window_size0.write((bit<32>)packet_counter_value, meta.tcp_window_size); //feature
                    udp_len0.write((bit<32>) packet_counter_value, meta.udp_len); //feature
                    icmp_type0.write((bit<32>) packet_counter_value, meta.icmp_type); //feature 
                    dst_port0.write((bit<32>) packet_counter_value, meta.dstPort); // lucid_pkt_id
                    src_port0.write((bit<32>) packet_counter_value, meta.srcPort); // lucid_pkt_id
                    src_ip0.write((bit<32>) packet_counter_value, hdr.ipv4.srcAddr); // lucid_pkt_id
                    dst_ip0.write((bit<32>) packet_counter_value, hdr.ipv4.dstAddr); // ludic_pkt_id
                    ip_upper_protocol0.write((bit<32>) packet_counter_value, hdr.ipv4.protocol); // lucid_pkt_id

                    bit<64> tmp;
                    not_foldable_counter0.read(tmp,0);
                    not_foldable_counter0.write(0,tmp+1);

                } // end block 0
                else {

                    bit<64> rc;
                    round_counter.read(rc,0);
                    log_msg("log_flow: {},{},{},{},{},{}",{meta.srcAddr,meta.srcPort, meta.dstAddr, meta.dstPort, meta.ipProtocol, rc});

                    packet_counter1.read(packet_counter_value,0);
                    packet_counter1.write(0, packet_counter_value+1);
                    bit<64> pim;
                    process_in_memory1.read(pim,0);
                    process_in_memory1.write(0,pim+1);
                    time1.write((bit<32>)packet_counter_value, meta.ingress_timestamp); //feature
                    packet_length1.write((bit<32>)packet_counter_value, meta.packet_length); //feature
                    ip_flags1.write((bit<32>) packet_counter_value, meta.ip_flags); //feature
                    tcp_len1.write((bit<32>) packet_counter_value, meta.tcp_len); //feature
                    tcp_ack1.write((bit<32>) packet_counter_value, meta.tcp_ack); //feature
                    tcp_flags1.write((bit<32>) packet_counter_value, meta.tcp_flags); //feature
                    tcp_window_size1.write((bit<32>)packet_counter_value, meta.tcp_window_size); //feature
                    udp_len1.write((bit<32>) packet_counter_value, meta.udp_len); //feature
                    icmp_type1.write((bit<32>) packet_counter_value, meta.icmp_type); //feature 
                    dst_port1.write((bit<32>) packet_counter_value, meta.dstPort); // lucid_pkt_id
                    src_port1.write((bit<32>) packet_counter_value, meta.srcPort); // lucid_pkt_id
                    src_ip1.write((bit<32>) packet_counter_value, hdr.ipv4.srcAddr); // lucid_pkt_id
                    dst_ip1.write((bit<32>) packet_counter_value, hdr.ipv4.dstAddr); // ludic_pkt_id
                    ip_upper_protocol1.write((bit<32>) packet_counter_value, hdr.ipv4.protocol); // lucid_pkt_id

                    bit<64> tmp;
                    not_foldable_counter1.read(tmp,0);
                    not_foldable_counter1.write(0,tmp+1);
                } // end block 1
            } // end sampling
        }// valid ipv4_address
    }// apply
}// MyIngress

/*************************************************************************
****************  E G R E S S   P R O C E S S I N G   *******************
*************************************************************************/

control MyEgress(inout headers hdr,
                 inout metadata meta,
                 inout standard_metadata_t standard_metadata) {
    apply { 
       
    }
}

/*************************************************************************
*************   C H E C K S U M    C O M P U T A T I O N   **************
*************************************************************************/

control MyComputeChecksum(inout headers  hdr, inout metadata meta) {
     apply {
	update_checksum(
	    hdr.ipv4.isValid(),
            { hdr.ipv4.version,
	      hdr.ipv4.ihl,
              hdr.ipv4.diffserv,
              hdr.ipv4.totalLen,
              hdr.ipv4.identification,
              hdr.ipv4.flags,
              hdr.ipv4.fragOffset,
              hdr.ipv4.ttl,
              hdr.ipv4.protocol,
              hdr.ipv4.srcAddr,
              hdr.ipv4.dstAddr },
            hdr.ipv4.hdrChecksum,
            HashAlgorithm.csum16);
    } 
}

/*************************************************************************
***********************  D E P A R S E R  *******************************
*************************************************************************/

control MyDeparser(packet_out packet, in headers hdr) {
    apply {
        packet.emit(hdr.ethernet);
        packet.emit(hdr.ipv4);
        packet.emit(hdr.icmp);
        packet.emit(hdr.udp);
        packet.emit(hdr.tcp);
        
    }
}

/*************************************************************************
***********************  S W I T C H  *******************************
*************************************************************************/

V1Switch(
MyParser(),
MyVerifyChecksum(),
MyIngress(),
MyEgress(),
MyComputeChecksum(),
MyDeparser()
) main;
