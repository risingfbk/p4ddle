{"payload":{"allShortcutsEnabled":false,"fileTree":{"tools":{"items":[{"name":"macos","path":"tools/macos","contentType":"directory"},{"name":".gitignore","path":"tools/.gitignore","contentType":"file"},{"name":"Makefile.am","path":"tools/Makefile.am","contentType":"file"},{"name":"bm_CLI.in","path":"tools/bm_CLI.in","contentType":"file"},{"name":"bm_nanomsg_events.in","path":"tools/bm_nanomsg_events.in","contentType":"file"},{"name":"bm_p4dbg.in","path":"tools/bm_p4dbg.in","contentType":"file"},{"name":"bmpy_utils.py","path":"tools/bmpy_utils.py","contentType":"file"},{"name":"check_style.sh","path":"tools/check_style.sh","contentType":"file"},{"name":"cpplint.py","path":"tools/cpplint.py","contentType":"file"},{"name":"get_version.sh","path":"tools/get_version.sh","contentType":"file"},{"name":"nanomsg_client.py","path":"tools/nanomsg_client.py","contentType":"file"},{"name":"p4dbg.py","path":"tools/p4dbg.py","contentType":"file"},{"name":"run_valgrind.sh","path":"tools/run_valgrind.sh","contentType":"file"},{"name":"runtime_CLI.py","path":"tools/runtime_CLI.py","contentType":"file"},{"name":"veth_setup.sh","path":"tools/veth_setup.sh","contentType":"file"},{"name":"veth_teardown.sh","path":"tools/veth_teardown.sh","contentType":"file"}],"totalCount":16},"":{"items":[{"name":".github","path":".github","contentType":"directory"},{"name":"PI","path":"PI","contentType":"directory"},{"name":"ci","path":"ci","contentType":"directory"},{"name":"docs","path":"docs","contentType":"directory"},{"name":"examples","path":"examples","contentType":"directory"},{"name":"hooks","path":"hooks","contentType":"directory"},{"name":"include","path":"include","contentType":"directory"},{"name":"m4","path":"m4","contentType":"directory"},{"name":"mininet","path":"mininet","contentType":"directory"},{"name":"pdfixed","path":"pdfixed","contentType":"directory"},{"name":"services","path":"services","contentType":"directory"},{"name":"src","path":"src","contentType":"directory"},{"name":"targets","path":"targets","contentType":"directory"},{"name":"tests","path":"tests","contentType":"directory"},{"name":"third_party","path":"third_party","contentType":"directory"},{"name":"thrift_src","path":"thrift_src","contentType":"directory"},{"name":"tools","path":"tools","contentType":"directory"},{"name":".dockerignore","path":".dockerignore","contentType":"file"},{"name":".gitignore","path":".gitignore","contentType":"file"},{"name":".gitmodules","path":".gitmodules","contentType":"file"},{"name":"CONTRIBUTING.md","path":"CONTRIBUTING.md","contentType":"file"},{"name":"CPPLINT.cfg","path":"CPPLINT.cfg","contentType":"file"},{"name":"Dockerfile","path":"Dockerfile","contentType":"file"},{"name":"Dockerfile.noPI","path":"Dockerfile.noPI","contentType":"file"},{"name":"Doxyfile","path":"Doxyfile","contentType":"file"},{"name":"Doxymain.md","path":"Doxymain.md","contentType":"file"},{"name":"LICENSE","path":"LICENSE","contentType":"file"},{"name":"Makefile.am","path":"Makefile.am","contentType":"file"},{"name":"README.md","path":"README.md","contentType":"file"},{"name":"VERSION","path":"VERSION","contentType":"file"},{"name":"autogen.sh","path":"autogen.sh","contentType":"file"},{"name":"configure.ac","path":"configure.ac","contentType":"file"},{"name":"install_deps.sh","path":"install_deps.sh","contentType":"file"},{"name":"install_deps_ubuntu_22.04.sh","path":"install_deps_ubuntu_22.04.sh","contentType":"file"}],"totalCount":34}},"fileTreeProcessingTime":6.213366,"foldersToFetch":[],"repo":{"id":29883110,"defaultBranch":"main","name":"behavioral-model","ownerLogin":"p4lang","currentUserCanPush":false,"isFork":false,"isEmpty":false,"createdAt":"2015-01-26T21:43:23.000Z","ownerAvatar":"https://avatars.githubusercontent.com/u/10765181?v=4","public":true,"private":false,"isOrgOwned":true},"symbolsExpanded":false,"treeExpanded":true,"refInfo":{"name":"main","listCacheKey":"v0:1704219076.0","canEdit":false,"refType":"branch","currentOid":"5f1c590c7bdb32ababb6d6fe18977cf13ae3b043"},"path":"tools/bmpy_utils.py","currentUser":null,"blob":{"rawLines":["#!/usr/bin/env python3","","# Copyright 2013-present Barefoot Networks, Inc.","#","# Licensed under the Apache License, Version 2.0 (the \"License\");","# you may not use this file except in compliance with the License.","# You may obtain a copy of the License at","#","#   http://www.apache.org/licenses/LICENSE-2.0","#","# Unless required by applicable law or agreed to in writing, software","# distributed under the License is distributed on an \"AS IS\" BASIS,","# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.","# See the License for the specific language governing permissions and","# limitations under the License.","#","","#","# Antonin Bas (antonin@barefootnetworks.com)","#","#","","import sys","import hashlib","","from thrift import Thrift","from thrift.transport import TSocket","from thrift.transport import TTransport","from thrift.protocol import TBinaryProtocol","from thrift.protocol import TMultiplexedProtocol","","","def check_JSON_md5(client, json_src, out=sys.stdout):","    with open(json_src, 'rb') as f:","        m = hashlib.md5()","        for L in f:","            m.update(L)","        md5sum = m.digest()","","    def my_print(s):","        out.write(s)","","    try:","        bm_md5sum = client.bm_get_config_md5()","    except:","        my_print(\"Error when requesting config md5 sum from switch\\n\")","        sys.exit(1)","","    if md5sum != bm_md5sum:","        my_print(\"**********\\n\")","        my_print(\"WARNING: the JSON files loaded into the switch and the one \")","        my_print(\"you just provided to this CLI don't have the same md5 sum. \")","        my_print(\"Are you sure they describe the same program?\\n\")","        bm_md5sum_str = \"\".join(\"{:02x}\".format(ord(c)) for c in bm_md5sum)","        md5sum_str = \"\".join(\"{:02x}\".format(ord(c)) for c in md5sum)","        my_print(\"{:<15}: {}\\n\".format(\"switch md5\", bm_md5sum_str))","        my_print(\"{:<15}: {}\\n\".format(\"CLI input md5\", md5sum_str))","        my_print(\"**********\\n\")","","","def get_json_config(standard_client=None, json_path=None, out=sys.stdout):","    def my_print(s):","        out.write(s)","","    if json_path:","        if standard_client is not None:","            check_JSON_md5(standard_client, json_path)","        with open(json_path, encoding=\"utf-8\") as f:","            return f.read()","    else:","        assert(standard_client is not None)","        try:","            my_print(\"Obtaining JSON from switch...\\n\")","            json_cfg = standard_client.bm_get_config()","            my_print(\"Done\\n\")","        except:","            my_print(\"Error when requesting JSON config from switch\\n\")","            sys.exit(1)","        return json_cfg","","# services is [(service_name, client_class), ...]","","","def thrift_connect(thrift_ip, thrift_port, services, out=sys.stdout):","    def my_print(s):","        out.write(s)","","    # Make socket","    transport = TSocket.TSocket(thrift_ip, thrift_port)","    # Buffering is critical. Raw sockets are very slow","    transport = TTransport.TBufferedTransport(transport)","    # Wrap in a protocol","    bprotocol = TBinaryProtocol.TBinaryProtocol(transport)","","    clients = []","","    for service_name, service_cls in services:","        if service_name is None:","            clients.append(None)","            continue","        protocol = TMultiplexedProtocol.TMultiplexedProtocol(","            bprotocol, service_name)","        client = service_cls(protocol)","        clients.append(client)","","    # Connect!","    try:","        transport.open()","    except TTransport.TTransportException:","        my_print(\"Could not connect to thrift client on port {}\\n\".format(","            thrift_port))","        my_print(\"Make sure the switch is running \")","        my_print(\"and that you have the right port\\n\")","        sys.exit(1)","","    return clients","","","def thrift_connect_standard(thrift_ip, thrift_port, out=sys.stdout):","    from bm_runtime.standard import Standard","    return thrift_connect(thrift_ip, thrift_port,","                          [(\"standard\", Standard.Client)], out)[0]"],"stylingDirectives":[[{"start":0,"end":22,"cssClass":"pl-c"}],[],[{"start":0,"end":48,"cssClass":"pl-c"}],[{"start":0,"end":1,"cssClass":"pl-c"}],[{"start":0,"end":65,"cssClass":"pl-c"}],[{"start":0,"end":66,"cssClass":"pl-c"}],[{"start":0,"end":41,"cssClass":"pl-c"}],[{"start":0,"end":1,"cssClass":"pl-c"}],[{"start":0,"end":46,"cssClass":"pl-c"}],[{"start":0,"end":1,"cssClass":"pl-c"}],[{"start":0,"end":69,"cssClass":"pl-c"}],[{"start":0,"end":67,"cssClass":"pl-c"}],[{"start":0,"end":74,"cssClass":"pl-c"}],[{"start":0,"end":69,"cssClass":"pl-c"}],[{"start":0,"end":32,"cssClass":"pl-c"}],[{"start":0,"end":1,"cssClass":"pl-c"}],[],[{"start":0,"end":1,"cssClass":"pl-c"}],[{"start":0,"end":44,"cssClass":"pl-c"}],[{"start":0,"end":1,"cssClass":"pl-c"}],[{"start":0,"end":1,"cssClass":"pl-c"}],[],[{"start":0,"end":6,"cssClass":"pl-k"},{"start":7,"end":10,"cssClass":"pl-s1"}],[{"start":0,"end":6,"cssClass":"pl-k"},{"start":7,"end":14,"cssClass":"pl-s1"}],[],[{"start":0,"end":4,"cssClass":"pl-k"},{"start":5,"end":11,"cssClass":"pl-s1"},{"start":12,"end":18,"cssClass":"pl-k"},{"start":19,"end":25,"cssClass":"pl-v"}],[{"start":0,"end":4,"cssClass":"pl-k"},{"start":5,"end":11,"cssClass":"pl-s1"},{"start":12,"end":21,"cssClass":"pl-s1"},{"start":22,"end":28,"cssClass":"pl-k"},{"start":29,"end":36,"cssClass":"pl-v"}],[{"start":0,"end":4,"cssClass":"pl-k"},{"start":5,"end":11,"cssClass":"pl-s1"},{"start":12,"end":21,"cssClass":"pl-s1"},{"start":22,"end":28,"cssClass":"pl-k"},{"start":29,"end":39,"cssClass":"pl-v"}],[{"start":0,"end":4,"cssClass":"pl-k"},{"start":5,"end":11,"cssClass":"pl-s1"},{"start":12,"end":20,"cssClass":"pl-s1"},{"start":21,"end":27,"cssClass":"pl-k"},{"start":28,"end":43,"cssClass":"pl-v"}],[{"start":0,"end":4,"cssClass":"pl-k"},{"start":5,"end":11,"cssClass":"pl-s1"},{"start":12,"end":20,"cssClass":"pl-s1"},{"start":21,"end":27,"cssClass":"pl-k"},{"start":28,"end":48,"cssClass":"pl-v"}],[],[],[{"start":0,"end":3,"cssClass":"pl-k"},{"start":4,"end":18,"cssClass":"pl-en"},{"start":19,"end":25,"cssClass":"pl-s1"},{"start":27,"end":35,"cssClass":"pl-s1"},{"start":37,"end":40,"cssClass":"pl-s1"},{"start":40,"end":41,"cssClass":"pl-c1"},{"start":41,"end":44,"cssClass":"pl-s1"},{"start":45,"end":51,"cssClass":"pl-s1"}],[{"start":4,"end":8,"cssClass":"pl-k"},{"start":9,"end":13,"cssClass":"pl-en"},{"start":14,"end":22,"cssClass":"pl-s1"},{"start":24,"end":28,"cssClass":"pl-s"},{"start":30,"end":32,"cssClass":"pl-k"},{"start":33,"end":34,"cssClass":"pl-s1"}],[{"start":8,"end":9,"cssClass":"pl-s1"},{"start":10,"end":11,"cssClass":"pl-c1"},{"start":12,"end":19,"cssClass":"pl-s1"},{"start":20,"end":23,"cssClass":"pl-en"}],[{"start":8,"end":11,"cssClass":"pl-k"},{"start":12,"end":13,"cssClass":"pl-v"},{"start":14,"end":16,"cssClass":"pl-c1"},{"start":17,"end":18,"cssClass":"pl-s1"}],[{"start":12,"end":13,"cssClass":"pl-s1"},{"start":14,"end":20,"cssClass":"pl-en"},{"start":21,"end":22,"cssClass":"pl-v"}],[{"start":8,"end":14,"cssClass":"pl-s1"},{"start":15,"end":16,"cssClass":"pl-c1"},{"start":17,"end":18,"cssClass":"pl-s1"},{"start":19,"end":25,"cssClass":"pl-en"}],[],[{"start":4,"end":7,"cssClass":"pl-k"},{"start":8,"end":16,"cssClass":"pl-en"},{"start":17,"end":18,"cssClass":"pl-s1"}],[{"start":8,"end":11,"cssClass":"pl-s1"},{"start":12,"end":17,"cssClass":"pl-en"},{"start":18,"end":19,"cssClass":"pl-s1"}],[],[{"start":4,"end":7,"cssClass":"pl-k"}],[{"start":8,"end":17,"cssClass":"pl-s1"},{"start":18,"end":19,"cssClass":"pl-c1"},{"start":20,"end":26,"cssClass":"pl-s1"},{"start":27,"end":44,"cssClass":"pl-en"}],[{"start":4,"end":10,"cssClass":"pl-k"}],[{"start":8,"end":16,"cssClass":"pl-en"},{"start":17,"end":69,"cssClass":"pl-s"},{"start":66,"end":68,"cssClass":"pl-cce"}],[{"start":8,"end":11,"cssClass":"pl-s1"},{"start":12,"end":16,"cssClass":"pl-en"},{"start":17,"end":18,"cssClass":"pl-c1"}],[],[{"start":4,"end":6,"cssClass":"pl-k"},{"start":7,"end":13,"cssClass":"pl-s1"},{"start":14,"end":16,"cssClass":"pl-c1"},{"start":17,"end":26,"cssClass":"pl-s1"}],[{"start":8,"end":16,"cssClass":"pl-en"},{"start":17,"end":31,"cssClass":"pl-s"},{"start":28,"end":30,"cssClass":"pl-cce"}],[{"start":8,"end":16,"cssClass":"pl-en"},{"start":17,"end":78,"cssClass":"pl-s"}],[{"start":8,"end":16,"cssClass":"pl-en"},{"start":17,"end":78,"cssClass":"pl-s"}],[{"start":8,"end":16,"cssClass":"pl-en"},{"start":17,"end":65,"cssClass":"pl-s"},{"start":62,"end":64,"cssClass":"pl-cce"}],[{"start":8,"end":21,"cssClass":"pl-s1"},{"start":22,"end":23,"cssClass":"pl-c1"},{"start":24,"end":26,"cssClass":"pl-s"},{"start":27,"end":31,"cssClass":"pl-en"},{"start":32,"end":40,"cssClass":"pl-s"},{"start":41,"end":47,"cssClass":"pl-en"},{"start":48,"end":51,"cssClass":"pl-en"},{"start":52,"end":53,"cssClass":"pl-s1"},{"start":56,"end":59,"cssClass":"pl-k"},{"start":60,"end":61,"cssClass":"pl-s1"},{"start":62,"end":64,"cssClass":"pl-c1"},{"start":65,"end":74,"cssClass":"pl-s1"}],[{"start":8,"end":18,"cssClass":"pl-s1"},{"start":19,"end":20,"cssClass":"pl-c1"},{"start":21,"end":23,"cssClass":"pl-s"},{"start":24,"end":28,"cssClass":"pl-en"},{"start":29,"end":37,"cssClass":"pl-s"},{"start":38,"end":44,"cssClass":"pl-en"},{"start":45,"end":48,"cssClass":"pl-en"},{"start":49,"end":50,"cssClass":"pl-s1"},{"start":53,"end":56,"cssClass":"pl-k"},{"start":57,"end":58,"cssClass":"pl-s1"},{"start":59,"end":61,"cssClass":"pl-c1"},{"start":62,"end":68,"cssClass":"pl-s1"}],[{"start":8,"end":16,"cssClass":"pl-en"},{"start":17,"end":31,"cssClass":"pl-s"},{"start":28,"end":30,"cssClass":"pl-cce"},{"start":32,"end":38,"cssClass":"pl-en"},{"start":39,"end":51,"cssClass":"pl-s"},{"start":53,"end":66,"cssClass":"pl-s1"}],[{"start":8,"end":16,"cssClass":"pl-en"},{"start":17,"end":31,"cssClass":"pl-s"},{"start":28,"end":30,"cssClass":"pl-cce"},{"start":32,"end":38,"cssClass":"pl-en"},{"start":39,"end":54,"cssClass":"pl-s"},{"start":56,"end":66,"cssClass":"pl-s1"}],[{"start":8,"end":16,"cssClass":"pl-en"},{"start":17,"end":31,"cssClass":"pl-s"},{"start":28,"end":30,"cssClass":"pl-cce"}],[],[],[{"start":0,"end":3,"cssClass":"pl-k"},{"start":4,"end":19,"cssClass":"pl-en"},{"start":20,"end":35,"cssClass":"pl-s1"},{"start":35,"end":36,"cssClass":"pl-c1"},{"start":36,"end":40,"cssClass":"pl-c1"},{"start":42,"end":51,"cssClass":"pl-s1"},{"start":51,"end":52,"cssClass":"pl-c1"},{"start":52,"end":56,"cssClass":"pl-c1"},{"start":58,"end":61,"cssClass":"pl-s1"},{"start":61,"end":62,"cssClass":"pl-c1"},{"start":62,"end":65,"cssClass":"pl-s1"},{"start":66,"end":72,"cssClass":"pl-s1"}],[{"start":4,"end":7,"cssClass":"pl-k"},{"start":8,"end":16,"cssClass":"pl-en"},{"start":17,"end":18,"cssClass":"pl-s1"}],[{"start":8,"end":11,"cssClass":"pl-s1"},{"start":12,"end":17,"cssClass":"pl-en"},{"start":18,"end":19,"cssClass":"pl-s1"}],[],[{"start":4,"end":6,"cssClass":"pl-k"},{"start":7,"end":16,"cssClass":"pl-s1"}],[{"start":8,"end":10,"cssClass":"pl-k"},{"start":11,"end":26,"cssClass":"pl-s1"},{"start":27,"end":29,"cssClass":"pl-c1"},{"start":30,"end":33,"cssClass":"pl-c1"},{"start":34,"end":38,"cssClass":"pl-c1"}],[{"start":12,"end":26,"cssClass":"pl-en"},{"start":27,"end":42,"cssClass":"pl-s1"},{"start":44,"end":53,"cssClass":"pl-s1"}],[{"start":8,"end":12,"cssClass":"pl-k"},{"start":13,"end":17,"cssClass":"pl-en"},{"start":18,"end":27,"cssClass":"pl-s1"},{"start":29,"end":37,"cssClass":"pl-s1"},{"start":37,"end":38,"cssClass":"pl-c1"},{"start":38,"end":45,"cssClass":"pl-s"},{"start":47,"end":49,"cssClass":"pl-k"},{"start":50,"end":51,"cssClass":"pl-s1"}],[{"start":12,"end":18,"cssClass":"pl-k"},{"start":19,"end":20,"cssClass":"pl-s1"},{"start":21,"end":25,"cssClass":"pl-en"}],[{"start":4,"end":8,"cssClass":"pl-k"}],[{"start":8,"end":14,"cssClass":"pl-k"},{"start":15,"end":30,"cssClass":"pl-s1"},{"start":31,"end":33,"cssClass":"pl-c1"},{"start":34,"end":37,"cssClass":"pl-c1"},{"start":38,"end":42,"cssClass":"pl-c1"}],[{"start":8,"end":11,"cssClass":"pl-k"}],[{"start":12,"end":20,"cssClass":"pl-en"},{"start":21,"end":54,"cssClass":"pl-s"},{"start":51,"end":53,"cssClass":"pl-cce"}],[{"start":12,"end":20,"cssClass":"pl-s1"},{"start":21,"end":22,"cssClass":"pl-c1"},{"start":23,"end":38,"cssClass":"pl-s1"},{"start":39,"end":52,"cssClass":"pl-en"}],[{"start":12,"end":20,"cssClass":"pl-en"},{"start":21,"end":29,"cssClass":"pl-s"},{"start":26,"end":28,"cssClass":"pl-cce"}],[{"start":8,"end":14,"cssClass":"pl-k"}],[{"start":12,"end":20,"cssClass":"pl-en"},{"start":21,"end":70,"cssClass":"pl-s"},{"start":67,"end":69,"cssClass":"pl-cce"}],[{"start":12,"end":15,"cssClass":"pl-s1"},{"start":16,"end":20,"cssClass":"pl-en"},{"start":21,"end":22,"cssClass":"pl-c1"}],[{"start":8,"end":14,"cssClass":"pl-k"},{"start":15,"end":23,"cssClass":"pl-s1"}],[],[{"start":0,"end":49,"cssClass":"pl-c"}],[],[],[{"start":0,"end":3,"cssClass":"pl-k"},{"start":4,"end":18,"cssClass":"pl-en"},{"start":19,"end":28,"cssClass":"pl-s1"},{"start":30,"end":41,"cssClass":"pl-s1"},{"start":43,"end":51,"cssClass":"pl-s1"},{"start":53,"end":56,"cssClass":"pl-s1"},{"start":56,"end":57,"cssClass":"pl-c1"},{"start":57,"end":60,"cssClass":"pl-s1"},{"start":61,"end":67,"cssClass":"pl-s1"}],[{"start":4,"end":7,"cssClass":"pl-k"},{"start":8,"end":16,"cssClass":"pl-en"},{"start":17,"end":18,"cssClass":"pl-s1"}],[{"start":8,"end":11,"cssClass":"pl-s1"},{"start":12,"end":17,"cssClass":"pl-en"},{"start":18,"end":19,"cssClass":"pl-s1"}],[],[{"start":4,"end":17,"cssClass":"pl-c"}],[{"start":4,"end":13,"cssClass":"pl-s1"},{"start":14,"end":15,"cssClass":"pl-c1"},{"start":16,"end":23,"cssClass":"pl-v"},{"start":24,"end":31,"cssClass":"pl-v"},{"start":32,"end":41,"cssClass":"pl-s1"},{"start":43,"end":54,"cssClass":"pl-s1"}],[{"start":4,"end":54,"cssClass":"pl-c"}],[{"start":4,"end":13,"cssClass":"pl-s1"},{"start":14,"end":15,"cssClass":"pl-c1"},{"start":16,"end":26,"cssClass":"pl-v"},{"start":27,"end":45,"cssClass":"pl-v"},{"start":46,"end":55,"cssClass":"pl-s1"}],[{"start":4,"end":24,"cssClass":"pl-c"}],[{"start":4,"end":13,"cssClass":"pl-s1"},{"start":14,"end":15,"cssClass":"pl-c1"},{"start":16,"end":31,"cssClass":"pl-v"},{"start":32,"end":47,"cssClass":"pl-v"},{"start":48,"end":57,"cssClass":"pl-s1"}],[],[{"start":4,"end":11,"cssClass":"pl-s1"},{"start":12,"end":13,"cssClass":"pl-c1"}],[],[{"start":4,"end":7,"cssClass":"pl-k"},{"start":8,"end":20,"cssClass":"pl-s1"},{"start":22,"end":33,"cssClass":"pl-s1"},{"start":34,"end":36,"cssClass":"pl-c1"},{"start":37,"end":45,"cssClass":"pl-s1"}],[{"start":8,"end":10,"cssClass":"pl-k"},{"start":11,"end":23,"cssClass":"pl-s1"},{"start":24,"end":26,"cssClass":"pl-c1"},{"start":27,"end":31,"cssClass":"pl-c1"}],[{"start":12,"end":19,"cssClass":"pl-s1"},{"start":20,"end":26,"cssClass":"pl-en"},{"start":27,"end":31,"cssClass":"pl-c1"}],[{"start":12,"end":20,"cssClass":"pl-k"}],[{"start":8,"end":16,"cssClass":"pl-s1"},{"start":17,"end":18,"cssClass":"pl-c1"},{"start":19,"end":39,"cssClass":"pl-v"},{"start":40,"end":60,"cssClass":"pl-v"}],[{"start":12,"end":21,"cssClass":"pl-s1"},{"start":23,"end":35,"cssClass":"pl-s1"}],[{"start":8,"end":14,"cssClass":"pl-s1"},{"start":15,"end":16,"cssClass":"pl-c1"},{"start":17,"end":28,"cssClass":"pl-en"},{"start":29,"end":37,"cssClass":"pl-s1"}],[{"start":8,"end":15,"cssClass":"pl-s1"},{"start":16,"end":22,"cssClass":"pl-en"},{"start":23,"end":29,"cssClass":"pl-s1"}],[],[{"start":4,"end":14,"cssClass":"pl-c"}],[{"start":4,"end":7,"cssClass":"pl-k"}],[{"start":8,"end":17,"cssClass":"pl-s1"},{"start":18,"end":22,"cssClass":"pl-en"}],[{"start":4,"end":10,"cssClass":"pl-k"},{"start":11,"end":21,"cssClass":"pl-v"},{"start":22,"end":41,"cssClass":"pl-v"}],[{"start":8,"end":16,"cssClass":"pl-en"},{"start":17,"end":66,"cssClass":"pl-s"},{"start":63,"end":65,"cssClass":"pl-cce"},{"start":67,"end":73,"cssClass":"pl-en"}],[{"start":12,"end":23,"cssClass":"pl-s1"}],[{"start":8,"end":16,"cssClass":"pl-en"},{"start":17,"end":51,"cssClass":"pl-s"}],[{"start":8,"end":16,"cssClass":"pl-en"},{"start":17,"end":53,"cssClass":"pl-s"},{"start":50,"end":52,"cssClass":"pl-cce"}],[{"start":8,"end":11,"cssClass":"pl-s1"},{"start":12,"end":16,"cssClass":"pl-en"},{"start":17,"end":18,"cssClass":"pl-c1"}],[],[{"start":4,"end":10,"cssClass":"pl-k"},{"start":11,"end":18,"cssClass":"pl-s1"}],[],[],[{"start":0,"end":3,"cssClass":"pl-k"},{"start":4,"end":27,"cssClass":"pl-en"},{"start":28,"end":37,"cssClass":"pl-s1"},{"start":39,"end":50,"cssClass":"pl-s1"},{"start":52,"end":55,"cssClass":"pl-s1"},{"start":55,"end":56,"cssClass":"pl-c1"},{"start":56,"end":59,"cssClass":"pl-s1"},{"start":60,"end":66,"cssClass":"pl-s1"}],[{"start":4,"end":8,"cssClass":"pl-k"},{"start":9,"end":19,"cssClass":"pl-s1"},{"start":20,"end":28,"cssClass":"pl-s1"},{"start":29,"end":35,"cssClass":"pl-k"},{"start":36,"end":44,"cssClass":"pl-v"}],[{"start":4,"end":10,"cssClass":"pl-k"},{"start":11,"end":25,"cssClass":"pl-en"},{"start":26,"end":35,"cssClass":"pl-s1"},{"start":37,"end":48,"cssClass":"pl-s1"}],[{"start":28,"end":38,"cssClass":"pl-s"},{"start":40,"end":48,"cssClass":"pl-v"},{"start":49,"end":55,"cssClass":"pl-v"},{"start":59,"end":62,"cssClass":"pl-s1"},{"start":64,"end":65,"cssClass":"pl-c1"}]],"csv":null,"csvError":null,"dependabotInfo":{"showConfigurationBanner":false,"configFilePath":null,"networkDependabotPath":"/p4lang/behavioral-model/network/updates","dismissConfigurationNoticePath":"/settings/dismiss-notice/dependabot_configuration_notice","configurationNoticeDismissed":null,"repoAlertsPath":"/p4lang/behavioral-model/security/dependabot","repoSecurityAndAnalysisPath":"/p4lang/behavioral-model/settings/security_analysis","repoOwnerIsOrg":true,"currentUserCanAdminRepo":false},"displayName":"bmpy_utils.py","displayUrl":"https://github.com/p4lang/behavioral-model/blob/main/tools/bmpy_utils.py?raw=true","headerInfo":{"blobSize":"3.79 KB","deleteInfo":{"deleteTooltip":"You must be signed in to make or propose changes"},"editInfo":{"editTooltip":"You must be signed in to make or propose changes"},"ghDesktopPath":"https://desktop.github.com","gitLfsPath":null,"onBranch":true,"shortPath":"600f074","siteNavLoginPath":"/login?return_to=https%3A%2F%2Fgithub.com%2Fp4lang%2Fbehavioral-model%2Fblob%2Fmain%2Ftools%2Fbmpy_utils.py","isCSV":false,"isRichtext":false,"toc":null,"lineInfo":{"truncatedLoc":"122","truncatedSloc":"100"},"mode":"file"},"image":false,"isCodeownersFile":null,"isPlain":false,"isValidLegacyIssueTemplate":false,"issueTemplateHelpUrl":"https://docs.github.com/articles/about-issue-and-pull-request-templates","issueTemplate":null,"discussionTemplate":null,"language":"Python","languageID":303,"large":false,"loggedIn":false,"newDiscussionPath":"/p4lang/behavioral-model/discussions/new","newIssuePath":"/p4lang/behavioral-model/issues/new","planSupportInfo":{"repoIsFork":null,"repoOwnedByCurrentUser":null,"requestFullPath":"/p4lang/behavioral-model/blob/main/tools/bmpy_utils.py","showFreeOrgGatedFeatureMessage":null,"showPlanSupportBanner":null,"upgradeDataAttributes":null,"upgradePath":null},"publishBannersInfo":{"dismissActionNoticePath":"/settings/dismiss-notice/publish_action_from_dockerfile","releasePath":"/p4lang/behavioral-model/releases/new?marketplace=true","showPublishActionBanner":false},"rawBlobUrl":"https://github.com/p4lang/behavioral-model/raw/main/tools/bmpy_utils.py","renderImageOrRaw":false,"richText":null,"renderedFileInfo":null,"shortPath":null,"symbolsEnabled":true,"tabSize":8,"topBannersInfo":{"overridingGlobalFundingFile":false,"globalPreferredFundingPath":null,"repoOwner":"p4lang","repoName":"behavioral-model","showInvalidCitationWarning":false,"citationHelpUrl":"https://docs.github.com/github/creating-cloning-and-archiving-repositories/creating-a-repository-on-github/about-citation-files","showDependabotConfigurationBanner":false,"actionsOnboardingTip":null},"truncated":false,"viewable":true,"workflowRedirectUrl":null,"symbols":{"timed_out":false,"not_analyzed":false,"symbols":[{"name":"check_JSON_md5","kind":"function","ident_start":901,"ident_end":915,"extent_start":897,"extent_end":1893,"fully_qualified_name":"check_JSON_md5","ident_utf16":{"start":{"line_number":32,"utf16_col":4},"end":{"line_number":32,"utf16_col":18}},"extent_utf16":{"start":{"line_number":32,"utf16_col":0},"end":{"line_number":57,"utf16_col":32}}},{"name":"my_print","kind":"function","ident_start":1094,"ident_end":1102,"extent_start":1090,"extent_end":1127,"fully_qualified_name":"my_print","ident_utf16":{"start":{"line_number":39,"utf16_col":8},"end":{"line_number":39,"utf16_col":16}},"extent_utf16":{"start":{"line_number":39,"utf16_col":4},"end":{"line_number":40,"utf16_col":20}}},{"name":"get_json_config","kind":"function","ident_start":1900,"ident_end":1915,"extent_start":1896,"extent_end":2552,"fully_qualified_name":"get_json_config","ident_utf16":{"start":{"line_number":60,"utf16_col":4},"end":{"line_number":60,"utf16_col":19}},"extent_utf16":{"start":{"line_number":60,"utf16_col":0},"end":{"line_number":78,"utf16_col":23}}},{"name":"my_print","kind":"function","ident_start":1979,"ident_end":1987,"extent_start":1975,"extent_end":2012,"fully_qualified_name":"my_print","ident_utf16":{"start":{"line_number":61,"utf16_col":8},"end":{"line_number":61,"utf16_col":16}},"extent_utf16":{"start":{"line_number":61,"utf16_col":4},"end":{"line_number":62,"utf16_col":20}}},{"name":"thrift_connect","kind":"function","ident_start":2610,"ident_end":2624,"extent_start":2606,"extent_end":3652,"fully_qualified_name":"thrift_connect","ident_utf16":{"start":{"line_number":83,"utf16_col":4},"end":{"line_number":83,"utf16_col":18}},"extent_utf16":{"start":{"line_number":83,"utf16_col":0},"end":{"line_number":115,"utf16_col":18}}},{"name":"my_print","kind":"function","ident_start":2684,"ident_end":2692,"extent_start":2680,"extent_end":2717,"fully_qualified_name":"my_print","ident_utf16":{"start":{"line_number":84,"utf16_col":8},"end":{"line_number":84,"utf16_col":16}},"extent_utf16":{"start":{"line_number":84,"utf16_col":4},"end":{"line_number":85,"utf16_col":20}}},{"name":"thrift_connect_standard","kind":"function","ident_start":3659,"ident_end":3682,"extent_start":3655,"extent_end":3885,"fully_qualified_name":"thrift_connect_standard","ident_utf16":{"start":{"line_number":118,"utf16_col":4},"end":{"line_number":118,"utf16_col":27}},"extent_utf16":{"start":{"line_number":118,"utf16_col":0},"end":{"line_number":121,"utf16_col":66}}}]}},"copilotInfo":null,"copilotAccessAllowed":false,"csrf_tokens":{"/p4lang/behavioral-model/branches":{"post":"HHFJEvKYYNs8QbXCuqoUxEWb4cYP3mz8GiiNCxO63jg2OEaI_GnV1p9YImimKvQOrMkLG13fKGpkcQOGM0wVLw"},"/repos/preferences":{"post":"t8VuVLoLLqsxnBnpIJC8F7sA5CRWrQtXtfRF7ZgFFbC915xYYQww0WRISWCXFLKodko7SLzFwLLOSIbE1J5jAA"}}},"title":"behavioral-model/tools/bmpy_utils.py at main · p4lang/behavioral-model"}