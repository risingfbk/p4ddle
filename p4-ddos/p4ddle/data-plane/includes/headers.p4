header ethernet_t {
    bit<48> dstAddr;
    bit<48> srcAddr;
    bit<16> etherType;
}

header ipv4_t {
    bit<4>  version;
    bit<4>  ihl;
    bit<8>  diffserv;
    bit<16> totalLen;
    bit<16> identification;
    bit<3>  flags;
    bit<13> fragOffset;
    bit<8>  ttl;
    bit<8>  protocol;
    bit<16> hdrChecksum;
    bit<32> srcAddr;
    bit<32> dstAddr;
}

header tcp_t{
    bit<16> srcPort;
    bit<16> dstPort;
    bit<32> seqNo;
    bit<32> ackNo;
    bit<4>  dataOffset;
    bit<3>  res;
    bit<9>  tcp_flags; 
    bit<16> window;
    bit<16> checksum; 
    bit<16> urgentPtr;
}

header udp_t {
    bit<16>  srcPort;
    bit<16>  dstPort;
    bit<16>  udplen;
    bit<16>  udpchk;
}

header icmp_t {
    bit<8>  type;
    bit<8>  code;
    bit<16> checksum;
}

struct headers {
    ethernet_t ethernet;
    ipv4_t     ipv4;
    tcp_t      tcp;
    udp_t      udp;
    icmp_t     icmp;
}

// ==================== LUCID ========================

struct metadata {
   // features
   bit<48> ingress_timestamp;
   bit<16> packet_length;
   bit<16> ip_flags;
   bit<16> tcp_len;
   bit<32> tcp_ack;
   bit<9> tcp_flags;
   bit<16> tcp_window_size;
   bit<16> udp_len;
   bit<8> icmp_type;

   //flow id
   bit<16> srcPort;
   bit<16> dstPort;
   bit<32> srcAddr;
   bit<32> dstAddr;
   bit<8>  ipProtocol;

   bit<32> index_fwd;
   bit<32> index_bwd;
   bit<32> index_fwd_short;
   bit<32> index_bwd_short;
}

