parser MyParser(packet_in packet,
                out headers hdr,
                inout metadata meta,
                inout standard_metadata_t standard_metadata) {
 
    state start {
        transition parse_ethernet;
    }

    state parse_ethernet {
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            0x800: parse_ipv4;
            default: accept;
        }
    }

    state parse_ipv4 {
        packet.extract(hdr.ipv4);

        meta.ingress_timestamp=standard_metadata.ingress_global_timestamp;
        meta.packet_length=hdr.ipv4.totalLen;
        meta.ip_flags=hdr.ipv4.flags++hdr.ipv4.fragOffset;
        meta.tcp_len=16w0;
        meta.tcp_ack=32w0;
        meta.tcp_flags=9w0;
        meta.tcp_window_size=16w0;
        meta.udp_len=16w0;
        meta.icmp_type=8w0;

        meta.srcPort=16w0;
        meta.dstPort=16w0;
        meta.srcAddr=hdr.ipv4.srcAddr;
        meta.dstAddr=hdr.ipv4.dstAddr;
        meta.ipProtocol=hdr.ipv4.protocol;

        transition select(hdr.ipv4.protocol) {
            8w0x1: parse_icmp;
            8w0x6: parse_tcp;
            8w0x11: parse_udp;
            default: accept;
        }
    }

    state parse_icmp {
        packet.extract(hdr.icmp);
        meta.icmp_type=hdr.icmp.type;

        transition accept;
    }
    
    state parse_tcp {
       packet.extract(hdr.tcp);

       bit<16> ip_header_len = ((bit<16>) hdr.ipv4.ihl) << 2; // x * 32 / 8 --> 1World=32 bit, but i need bytes
       bit<16> tcp_header_len = ((bit<16>) hdr.tcp.dataOffset) << 2; // x * 32 / 8 --> 1World=32 bit, but i need bytes
       meta.tcp_len=hdr.ipv4.totalLen - ip_header_len - tcp_header_len;

       meta.tcp_ack=hdr.tcp.ackNo;
       meta.tcp_flags=hdr.tcp.tcp_flags;
       meta.tcp_window_size=hdr.tcp.window;

       meta.srcPort=hdr.tcp.srcPort;
       meta.dstPort=hdr.tcp.dstPort;

       transition accept;
    }

    state parse_udp {
       packet.extract(hdr.udp);
       meta.udp_len=hdr.udp.udplen;
       meta.srcPort=hdr.udp.srcPort;
       meta.dstPort=hdr.udp.dstPort;
       transition accept;
    }

}
