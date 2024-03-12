// ==================  FIRST BLOCK =============================

// 9 features for LUCID
register<bit<48>>(PACKETS) time0;
register<bit<16>>(PACKETS) packet_length0;
register<bit<16>>(PACKETS) ip_flags0;
register<bit<16>>(PACKETS) tcp_len0; //16 xk la lunghezza di ip è 16 bit, tcp, ma non so quanto meno
register<bit<32>>(PACKETS) tcp_ack0;
register<bit<9>>(PACKETS) tcp_flags0;
register<bit<16>>(PACKETS) tcp_window_size0;
register<bit<16>>(PACKETS) udp_len0;
register<bit<8>>(PACKETS) icmp_type0;

// registers for LUCID internal behaviour
register<bit<16>>(PACKETS) dst_port0;
register<bit<16>>(PACKETS) src_port0;
register<bit<32>>(PACKETS) src_ip0;
register<bit<32>>(PACKETS) dst_ip0;
register<bit<8>>(PACKETS) ip_upper_protocol0;
register<bit<PACKET_COUNTER_WIDTH>>(1) packet_counter0; // counter from the position of a packets inside above registers

// ================== SECOND BLOCK =============================

// 9 features for LUCID
register<bit<48>>(PACKETS) time1;
register<bit<16>>(PACKETS) packet_length1;
register<bit<16>>(PACKETS) ip_flags1;
register<bit<16>>(PACKETS) tcp_len1; //16 xk la lunghezza di ip è 16 bit, tcp, ma non so quanto meno
register<bit<32>>(PACKETS) tcp_ack1;
register<bit<9>>(PACKETS) tcp_flags1;
register<bit<16>>(PACKETS) tcp_window_size1;
register<bit<16>>(PACKETS) udp_len1;
register<bit<8>>(PACKETS) icmp_type1;

// registers for LUCID internal behaviour
register<bit<16>>(PACKETS) dst_port1;
register<bit<16>>(PACKETS) src_port1;
register<bit<32>>(PACKETS) src_ip1;
register<bit<32>>(PACKETS) dst_ip1;
register<bit<8>>(PACKETS) ip_upper_protocol1;
register<bit<PACKET_COUNTER_WIDTH>>(1) packet_counter1; // counter from the position of a packets inside above registers

// ======================= OTHER REGISTERS ===================

register<bit<1>>(1) block_of_registers; // decide witch block of registers to use <REGISTERS>0 or <REGISTERS>1
register<bit<SAMPLING_COUNTER_WIDTH>>(1) sampling_counter; // counter for sampling, each time counter has the value of sampling_treshold the packets features are stored in features registers
register<bit<SAMPLING_COUNTER_WIDTH>>(1) sampling_treshold; // counter to track sampling number. i.e. treshold=2 means take one packet over two. Default is 1, initialized in commandsX.txt

register<bit<64>>(1) round_counter;//just for test

register<bit<64>>(1) not_foldable_counter0;//just for test
register<bit<64>>(1) process_in_memory0;//just for test

register<bit<64>>(1) not_foldable_counter1;//just for test
register<bit<64>>(1) process_in_memory1;//just for test

register<bit<64>>(1) num_packets;//just for test