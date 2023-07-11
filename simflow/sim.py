import configparser, re
import random
import socket, struct, os
import zlib
import pandas as pd
from scipy.stats import uniform, lognorm
import numpy as np

def number_of_packets(protocol, value):

    # ref https://www.sciencedirect.com/science/article/pii/S0140366420320223

    others_dist = [
        [0.3050265769901237, 'uniform', [0, 1]], 
        [0.5534464570342856, 'uniform', [0, 2]], 
        [0.6171070936758667, 'uniform', [0, 3]], 
        [0.6663235933351954, 'uniform', [0, 4]], 
        [0.6756391849981327, 'uniform', [0, 5]], 
        [0.7578139265700051, 'uniform', [0, 6]], 
        [0.8890776763951536, 'lognorm', [0.5207023493412835, 0, 7.8055992790704085]], 
        [0.9623638306125883, 'lognorm', [0.7701056575379265, 0, 22.10972501544739]], 
        [0.9916529570997478, 'lognorm', [1.1252645297342514, 0, 128.6451515069839]], 
        [0.9999999999999986, 'lognorm', [1.98383694524085, 0, 1084.4707584768782]]
        ]

    tcp_dist = [
        [0.058043754929076714, 'uniform', [0, 1]],
        [0.3685687469118785, 'uniform', [0, 2]],
        [0.42366606349920644, 'uniform', [0, 3]],
        [0.471420106931141, 'uniform', [0, 4]],
        [0.47730263513630683, 'uniform', [0, 5]],
        [0.6025937639832731, 'uniform', [0, 6]],
        [0.8091370473961513, 'lognorm', [0.5122381561489622, 0, 7.9543319950074585]],
        [0.9310328843526378, 'lognorm', [0.7507627341686453, 0, 21.74157327774004]],
        [0.9835403206883468, 'lognorm', [1.1085442272126311, 0, 115.96889383386637]], 
        [0.9999999999999979, 'lognorm', [1.9932118957877463, 0, 845.9121788017795]]
        ]

    udp_dist = [
        [0.5827340809061519, 'uniform', [0, 1]],
        [0.7699599480750471, 'uniform', [0, 2]],
        [0.8614304718025192, 'uniform', [0, 3]],
        [0.9176539904552028, 'uniform', [0, 4]],
        [0.9471015330137897, 'lognorm', [0.48946755142877785, 0, 9.674436486545002]],
        [0.9873293706985831, 'lognorm', [0.26026467622698696, 0, 4.8838363075201485]],
        [0.9920519234863422, 'lognorm', [0.44942994700845956, 0, 29.41090858057708]], 
        [0.9976831870953078, 'lognorm', [0.9247173397792914, 0, 67.48121917707935]],
        [0.9994950142513823, 'lognorm', [1.0498087651873083, 0, 594.4921878194859]],
        [0.9998157844216482, 'lognorm', [1.8547729167446296, 0, 7892.503625889563]],
        [0.9999722837380661, 'lognorm', [1.1263636801013632, 0, 19824.000189566883]],
        [0.9999998509076339, 'lognorm', [0.40128970829397104, 0, 90369.61088154344]],
        [0.9999999999999998, 'lognorm', [0.5733877781565124, 0, 12757029.472787634]]
    ]

    if protocol == socket.IPPROTO_UDP:
        for i in udp_dist:
            if i[0] >= value:
                if i[1] == "uniform":
                    return int(uniform(*i[2]).rvs()+1)
                else:
                    return int(lognorm(*i[2]).rvs()+1)
    elif protocol == socket.IPPROTO_TCP:
        for i in tcp_dist:
            if i[0] >= value:
                if i[1] == "uniform":
                    return int(uniform(*i[2]).rvs()+1)
                else:
                    return int(lognorm(*i[2]).rvs()+1)
    else:
        for i in others_dist:
            if i[0] >= value:
                if i[1] == "uniform":
                    return int(uniform(*i[2]).rvs()+1)
                else:
                    return int(lognorm(*i[2]).rvs()+1)

class StoringAlgorithm:
    def __init__(self):
        self.name = "StoringAlgorithm"
        self.stats = pd.DataFrame(columns=["num_packets","num_flows","identified_flows","quality","packets","limit_packets","total_packets","collisions","discarded"])

    def __str__(self):
        return self.name
    
    def getStatString(self):
        return ""
    
    def saveStats(self,folder):
        self.stats.to_csv(folder+"/"+self.filename+".csv",index=False)

    @staticmethod
    def flows_in_memory(memory):
        flows = list()
        numberOfFlows = list()
        for element in memory:
            if element != (0,0):           
                if element not in flows:
                    flows.append(element)
                    numberOfFlows.append(1)
                else:
                    numberOfFlows[flows.index(element)] += 1
        return len(flows), [ [frozenset(flows[i]),numberOfFlows[i]] for i in range(0,len(flows))]


class BloomFilter(StoringAlgorithm):
    def __init__(self, packets, bloomSize, sampling, flowSize, maxFlows):
        super().__init__()
        self.name = "BloomFilter"
        self.packets = packets
        self.bloomSize = bloomSize
        self.sampling = sampling
        self.flowSize = flowSize
        self.maxFlows = maxFlows
        self.filename = "bloom-%dp-%df-%db-%ds-%dh"%(packets,maxFlows,bloomSize,sampling,flowSize)
    
    def getStatString(self):
        return "%d Packets - %d Bloom Size - %d Sampling - %d Flow Size - %d Max Flows"%(self.packets, self.bloomSize, self.sampling, self.flowSize, self.maxFlows)

    def run(self, packets, flows):

        MEMORY_BLOOM = [ (0,0) for i in range(0,self.packets) ]
        MEMORY_BLOOM_INDEX = [ 0 for i in range(0,self.bloomSize) ]
        MEMORY_BLOOM_INDEX = [ 0 for i in range(0,self.bloomSize) ]
        MEMORY_BLOOM_COUNT = 0
        BLOOM_COLLISIONS_PACKETS = 0
        BLOOM_IGNORED_PACKETS = 0
        BLOOM_COLLECTED_FLOWS = list()
        BLOOM_IGNORED_FLOWS = list()

        for index, packet in enumerate(packets):

            if index % self.sampling == 0:

                fwd_flow_h = zlib.crc32(packet[0]) % self.bloomSize
                bwd_flow_h = zlib.crc32(packet[1]) % self.bloomSize
                fwd_flow_hash_r = zlib.crc32(packet[0][:-1]) % self.bloomSize
                bwd_flow_hash_r = zlib.crc32(packet[1][:-1]) % self.bloomSize

                MEMORY_BLOOM_INDEX[fwd_flow_h] += 1
                MEMORY_BLOOM_INDEX[bwd_flow_h] += 1
                MEMORY_BLOOM_INDEX[fwd_flow_hash_r] += 1
                MEMORY_BLOOM_INDEX[bwd_flow_hash_r] += 1
                
                if min(MEMORY_BLOOM_INDEX[fwd_flow_h], MEMORY_BLOOM_INDEX[bwd_flow_h],MEMORY_BLOOM_INDEX[fwd_flow_hash_r], MEMORY_BLOOM_INDEX[bwd_flow_hash_r]) <= self.flowSize:
                    MEMORY_BLOOM[MEMORY_BLOOM_COUNT%self.packets] = (packet[0],packet[1])
                    BLOOM_COLLECTED_FLOWS.append((packet[0],packet[1]))
                    MEMORY_BLOOM_COUNT += 1
                    if MEMORY_BLOOM_COUNT >= self.packets:
                        BLOOM_COLLISIONS_PACKETS += 1
                else:
                    BLOOM_IGNORED_PACKETS += 1
                    BLOOM_IGNORED_FLOWS.append((packet[0],packet[1]))

        size, elements = self.flows_in_memory(MEMORY_BLOOM)

        df_1 = pd.DataFrame(flows, columns=["flow","max_packets","total_packets"])
        df_2 = pd.DataFrame(elements, columns=["flow","packets"])
        df = pd.merge(df_1,df_2, on="flow", how="left")
        df["packets"].fillna(0, inplace=True)

        try:
            df['quality'] = df.apply(lambda x: x["packets"]/x["max_packets"], axis=1)
        except:
            print(df[df.max_packets < 1])

        self.stats.loc[len(self.stats.index)] = [len(packets),len(flows),size,df['quality'].mean(),df['packets'].mean(),df["max_packets"].mean(),df["total_packets"].mean(),len(set(BLOOM_IGNORED_FLOWS) - set(BLOOM_COLLECTED_FLOWS)),len(set(BLOOM_COLLECTED_FLOWS) - set(MEMORY_BLOOM))]


class PacketBased(StoringAlgorithm):
    def __init__(self, packets, sampling, flowSize, maxFlows):
        super().__init__()
        self.name = "PacketBased"
        self.packets = packets
        self.sampling = sampling
        self.flowSize = flowSize
        self.maxFlows = maxFlows
        self.filename = "packet-%dp-%df-%ds-%dh"%(packets,maxFlows,sampling,flowSize)

    def getStatString(self):
        return "%d Packets - %d Sampling - %d Flow Size - %d Max Flows"%(self.packets, self.sampling, self.flowSize, self.maxFlows)

    def run(self, packets, flows):
        MEMORY_PACKET = [ (0,0) for i in range(0,self.packets) ]
        MEMORY_PACKET_COUNT = 0
        PACKET_COLLISIONS_PACKETS = 0
        PACKET_COLLECTED_FLOWS = list()

        for index, packet in enumerate(packets):

            if index % self.sampling == 0:
                PACKET_COLLECTED_FLOWS.append((packet[0],packet[1]))
                MEMORY_PACKET[MEMORY_PACKET_COUNT%self.packets] = (packet[0],packet[1])
                MEMORY_PACKET_COUNT += 1

                if MEMORY_PACKET_COUNT >= self.packets:
                    PACKET_COLLISIONS_PACKETS += 1

        size, elements = self.flows_in_memory(MEMORY_PACKET)

        df_1 = pd.DataFrame(flows, columns=["flow","max_packets","total_packets"])
        df_2 = pd.DataFrame(elements, columns=["flow","packets"])
        df = pd.merge(df_1,df_2, on="flow", how="left")
        df["packets"].fillna(0, inplace=True)

        df['quality'] = df.apply(lambda x: 1 if x["packets"]/x["max_packets"] > 1 else x["packets"]/x["max_packets"], axis=1)

        self.stats.loc[len(self.stats.index)] = [len(packets),len(flows),size,df['quality'].mean(),df['packets'].mean(),df["max_packets"].mean(),df["total_packets"].mean(),0,len(set(PACKET_COLLECTED_FLOWS) - set(MEMORY_PACKET))]

def create_random_flows(numFlows, flowSize):

    # ref https://www.sciencedirect.com/science/article/pii/S0140366420320223

    protoValues = [socket.IPPROTO_ICMP,socket.IPPROTO_UDP,socket.IPPROTO_TCP]
    protoWeights = [3.06,43.09,53.85]

    numGenFlows = random.randint(1,numFlows)
    flows = list()
    packets = list()

    for i in range(0,numGenFlows):
        srcAddr = 2886729728 + random.randint(1,4096)
        dstAddr = 2886729728 + random.randint(1,4096)
        proto = random.choices(protoValues, protoWeights)[0]
        if proto == socket.IPPROTO_ICMP:
            srcPort = 0
            dstPort = 0
        else:
            srcPort = random.randint(1,65535)
            dstPort = random.randint(1,65535)
        fwdFlow = struct.pack("!IHIHB", srcAddr, srcPort, dstAddr, dstPort, proto)
        bwdFlow = struct.pack("!IHIHB", dstAddr, dstPort, srcAddr, srcPort, proto) 
        numGenPackets = number_of_packets(proto, random.random())
        if numGenPackets > flowSize:
            maxPackets = flowSize
        else:
            maxPackets = numGenPackets
        flows.append([frozenset([fwdFlow,bwdFlow]),maxPackets,numGenPackets])
        for j in range(0,numGenPackets):
            packets.append([fwdFlow,bwdFlow])

    return flows, random.sample(packets, len(packets))

def main():
    try:
        config = configparser.ConfigParser()
        cwd = os.path.dirname(__file__)
        config.read(cwd+"/sim.ini")

        random.seed(config.getint("DEFAULT", "RandomSeed", fallback=None))
        np.random.seed(seed=config.getint("DEFAULT", "RandomSeed", fallback=None))
        scenarioConfig = config["Scenario"]

        algorithms = list()
        for storingAlgorithm in re.split(", |,", scenarioConfig.get("StoringAlgorithms")):
            for memorySize in re.split(", |,", scenarioConfig.get("MemorySizes")): 
                for flowSize in re.split(", |,", scenarioConfig.get("MaxFlowSizes")):
                    for samplingRate in re.split(", |,", scenarioConfig.get("SamplingRates")):
                        for flows in re.split(", |,", scenarioConfig.get("Flows")):
                            if storingAlgorithm == "Packet":
                                algorithms.append(PacketBased(int(memorySize),int(samplingRate), int(flowSize), int(flows)))
                            elif storingAlgorithm == "BloomFilter":
                                    for bloomFilterSize in re.split(", |,", config.get("Algorithm.BloomFilter", "BloomFilterSizes")):
                                        algorithms.append(BloomFilter(int(memorySize), int(bloomFilterSize), int(samplingRate), int(flowSize), int(flows)))

        for round in range(0,scenarioConfig.getint("NumberOfRounds")):
            for alg in algorithms:
                print("Running: Round %d - Alg %s - %s" % (round, alg, alg.getStatString()))

                flows, packets = create_random_flows(scenarioConfig.getint("Flows"),flowSize=alg.flowSize)
                
                alg.run(packets,flows)

        logFolder = cwd + "/" + config.get("DEFAULT", "LogFolder")
        if not os.path.exists(logFolder):
            os.makedirs(logFolder)
        for alg in algorithms:
            print("Saving Log: %s - %s" % (alg,alg.getStatString()))
            alg.saveStats(logFolder)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
