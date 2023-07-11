### Compile
```sh
p4c --target bmv2 --arch v1model --std p4-16 p4_packet_management.p4
```
### Run
```
sudo python launcher.py --behavioral-exe simple_switch --json p4_packet_management.json --cli simple_switch_CLI
```
