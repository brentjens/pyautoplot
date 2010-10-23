from socket import gethostname
from numpy import floor

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
        raise ValueError(node_name +' is not a storage node or compute nodeode and does therefore not belong to any subcluster')




def get_node_number_in_subcluster(node_name=gethostname()):
    if is_storage_node(node_name):
        nodes_per_subcluster = 3
    elif is_compute_node(node_name):
        nodes_per_subcluster = 9
    else:
        raise ValueError(node_name +' is not a storage node or compute nodeode and does therefore not belong to any subcluster')
    return get_node_number(node_name)-nodes_per_subcluster*(get_subcluster_number(node_name) -1) -1



def get_storage_node_names(subcluster_number):
    return ['lse'+str(i+(subcluster_number-1)*3).rjust(3,'0') for i in [1,2,3]]



def get_data_dirs(subcluster_number):
    storage_nodes=get_storage_node_names(subcluster_number)
    return ['/net/sub%s/%s/data%s/' %(subcluster_number, lse, data) for lse in storage_nodes for data in range(1,5)]



def find_msses(msname):
    result=[]
    for directory in get_data_dirs(get_subcluster_number()):
        result += [s.strip() for s in os.popen("find %s/%s -iname '*.MS'"%(directory,msname), 'r')]
    return result



def find_my_msses(msname):
    msnames=find_msses(msname)
    n_msnames=len(msses)
    n = int(ceil(float(n_msnames)/num_proc))
    proc_id=get_node_number_in_subcluster()
    msses_here = msses[proc_id*n:min((proc_id+1)*n, n_msnames)]
    return msses_here

