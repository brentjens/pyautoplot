from socket import gethostname
from numpy import floor,array
import os


def is_compute_node(name=gethostname()):
    return len(name) == 6 and name[:3]=='lce' and name[3:].isdigit()


def is_storage_node(name=gethostname()):
    return len(name) == 6 and name[:3]=='lse' and name[3:].isdigit()


def is_frontend_node(name=gethostname()):
    return len(name) == 6 and name[:3]=='lfe' and name[3:].isdigit()



def get_node_number(name=gethostname()):
    return int(name[3:])



def get_subcluster_number(node_name=gethostname()):
    if is_compute_node(node_name):
        return int(floor((get_node_number(node_name)-1)/9)+1)
    elif is_storage_node(node_name):
        return int(floor((get_node_number(node_name)-1)/3)+1)
    elif is_frontend_node(node_name):
        raise ValueError(node_name +' is a frontend node and does not belong to any particular sub cluster')
    else:
        raise ValueError(node_name +' is not a storage node or compute node and does therefore not belong to any subcluster')




def get_node_number_in_subcluster(node_name=gethostname()):
    if is_storage_node(node_name):
        nodes_per_subcluster = 3
    elif is_compute_node(node_name):
        nodes_per_subcluster = 9
    else:
        raise ValueError(node_name +' is not a storage node or compute node and does therefore not belong to any subcluster')
    return get_node_number(node_name)-nodes_per_subcluster*(get_subcluster_number(node_name) -1) -1



def get_storage_node_names(subcluster_number):
    return ['lse'+str(i+(subcluster_number-1)*3).rjust(3,'0') for i in [1,2,3]]



def get_data_dirs(subcluster_number, root='/net'):
    storage_nodes=get_storage_node_names(subcluster_number)
    return [os.path.join(root, 'sub%s/%s/data%s/' %(subcluster_number, lse, data)) for lse in storage_nodes for data in range(1,5)]



def find_msses(sas_id, root='/net', node_name=gethostname()):
    result=[]
    for directory in get_data_dirs(get_subcluster_number(node_name=node_name), root=root):
        names=sorted(filter(lambda n: n.find(str(sas_id))>=0, os.popen('ls %s' % (directory,))))
        non_empty_names=[n for n in names if n.strip() != '']
        if len(non_empty_names) > 0:
            l = [len(n.strip()) for n in non_empty_names]
            min_l = min(l)
            name=non_empty_names[l.index(min_l)]
            result += [os.path.normpath(s.strip()) for s in os.popen("find %s/%s -iname '*.MS' 2>&1|grep -ve 'No such file or directory'"%(directory,name.strip()), 'r')]
            pass
    return result



def find_my_msses(msname, root='/net', node_name=gethostname(), nodes_per_subcluster=9):
    msnames=find_msses(msname, root=root, node_name=node_name)
    proc_id=get_node_number_in_subcluster(node_name=node_name)
    return msnames[proc_id::nodes_per_subcluster]


