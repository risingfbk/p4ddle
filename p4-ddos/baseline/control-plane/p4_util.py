import sys
sys.path.append('/usr/local/lib/python3.9/site-packages') # workaround for bmpy_utils and bm_runtime. these are not pip installations but come with behavioural model installation
import json
import bmpy_utils as utils
from collections import OrderedDict

from bm_runtime.standard import Standard
from bm_runtime.standard.ttypes import *
try:
    from bm_runtime.simple_pre import SimplePre
except:
    pass
try:
    from bm_runtime.simple_pre_lag import SimplePreLAG
except:
    pass


# name of p4 registers
LUCID_REGISTERS = ['time','packet_length','ip_flags', 'tcp_len', 'tcp_ack', 'tcp_flags', 'tcp_window_size', 'udp_len', 'icmp_type', 'dst_port', 'src_port', 'dst_ip', 'src_ip', 'ip_upper_protocol'] # THIS ORDER IS VERY IMPORTANT! KEEP IT
PACKET_COUNTER = 'packet_counter'
SAMPLING_COUNTER='sampling_counter'
BLOCK_OF_REGISTERS='block_of_registers'
ROUND_COUNTER='round_counter'

class RuntimeAPI():
    
    def __init__(self, pre_type, standard_client, device_config, mc_client=None):
      
        self.client = standard_client
        self.mc_client = mc_client
        self.pre_type = pre_type
        self.device_config = device_config
        self.features = LUCID_REGISTERS


    @staticmethod
    def get_thrift_services(pre_type):
        services = [("standard", Standard.Client)]

        if pre_type == 1: # PreType.SimplePre:
            services += [("simple_pre", SimplePre.Client)]
        elif pre_type == 2: # PreType.SimplePreLAG:
            services += [("simple_pre_lag", SimplePreLAG.Client)]
        else:
            services += [(None, None)]

        return services
    
    
    def __check_field(self,field_type,field_name):
        try:
            self.device_config[field_type][field_name]
        except KeyError:
            raise UIn_ResourceError(field_type, field_name)
	
    def read_features_registers(self, block):
        """
         return list of lists in this format: [[feature1_val1,feature1_val2, ... ], [feature2_val1, feature2_val2, ...], ... ]
        """
        results = list()

        splitter_value=self.read_register(PACKET_COUNTER+str(block),0)
        for feature in self.features:
            entries=self.read_register(feature+str(block))
            # sort values based on current PACKET_COUNTER position
            entries=entries[splitter_value:]+entries[0:splitter_value]
            results.append(entries)
        return results
       
    def read_register(self, name,  index=None):
        "Read register value: register_read <name> [index]"
        register_name = name
        register_index = index

        try:
            self.__check_field("register",register_name)
                
            if register_index is not None:
                try:
                    register_index = int(register_index)
                except:
                    raise UIn_Error("Bad format for index")
                value = self.client.bm_register_read(0, register_name, index)
                # print("{}[{}]=".format(register_name, index), value)
                return value
            else:
              #  sys.stderr.write("register index omitted, reading entire array\n")
                entries = self.client.bm_register_read_all(0, name)

              #  print("{}=".format(register_name), ", ".join( [str(e) for e in entries]))
                return entries

        except UIn_Error as e:
            print(e)


    def write_register(self, name,  value, index=0):
        register_name = name
        register_index = index

        try:
            self.__check_field("register",register_name)
 
            try:
                register_index = int(register_index)
            except:
                raise UIn_Error("Bad format for index")
            self.client.bm_register_write(0, register_name, register_index, value)

        except UIn_Error as e:
            print(e)
            

    def reset_all_registers(self, block=None):
        if block is not None:
            for feature in self.features:
                self.reset_register(feature+str(block))
            self.reset_register(PACKET_COUNTER+str(block))
        else:
            self.reset_register(SAMPLING_COUNTER)
            for feature in self.features:
                self.reset_register(feature+"0")
                self.reset_register(feature+"1")
            self.reset_register(PACKET_COUNTER+"0")
            self.reset_register(PACKET_COUNTER+"1")
            self.reset_register(BLOCK_OF_REGISTERS)

    def reset_register(self, name):
        register_name = name
        try:
            self.__check_field("register",register_name)
            self.client.bm_register_reset(0, name)
           # print("reset", name)
        except UIn_Error as e:
            print(e)



    def switch_register_block(self):
        """
          Switch register block, so when you are reading one block bmv2 will write on the other one
        """
        current_block=self.read_register(BLOCK_OF_REGISTERS,0)
        current_round=self.read_register(ROUND_COUNTER,0)

        if current_block == 0:
            self.write_register(BLOCK_OF_REGISTERS,1,0)
        else:
            self.write_register(BLOCK_OF_REGISTERS,0,0)

        self.write_register(ROUND_COUNTER,current_round+1,0)

        return current_block
        

class UIn_Error(Exception):
    def __init__(self, info=""):
        self.info = info

    def __str__(self):
        return self.info

class UIn_ResourceError(UIn_Error):
    def __init__(self, res_type, name):
        self.res_type = res_type
        self.name = name

    def __str__(self):
        return "Invalid %s name (%s)" % (self.res_type, self.name)



class Register:
    def __init__(self, name, id_):
        self.name = name
        self.id_ = id_
        self.width = None
        self.size = None
    def register_str(self):
        return "{0:30} [{1}]".format(self.name, self.size)




def load_json_str(json_str, architecture_spec=None):

    def get_json_key(key):
        return json_.get(key, [])

    device_config=dict()
    registers=dict()

    json_ = json.loads(json_str)

    
    for j_register in get_json_key("register_arrays"):
        register = Register(j_register["name"], j_register["id"])
        register.size = j_register["size"]
        register.width = j_register["bitwidth"]
        registers[register.name]=register

    device_config["register"]=registers
  
    return device_config





def load_json_config(standard_client=None, json_path=None, architecture_spec=None):
    """
    Obtaining json from swich if json_path is None
    """
    return load_json_str(utils.get_json_config(
        standard_client, json_path), architecture_spec)









def get_runtime_API(host,port,pre,json):


    standard_client, mc_client = utils.thrift_connect(
        host, port,
        RuntimeAPI.get_thrift_services(pre)
    )
    device_config = load_json_config(standard_client, json)

    return RuntimeAPI(pre, standard_client, device_config, mc_client)
