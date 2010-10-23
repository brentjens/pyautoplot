#!/usr/bin/env python
import os
from pyautoplot.lofaroffline import *
from numpy import ceil

"""
Create the offline storage cluster file system structure in
testdata/net/... and add a test MS to it.
"""

def ensure_dir(path):
    try:
        os.mkdir(path)
    except (OSError,) as e:
        pass
    
def make_empty_file(path):
    return open(path, mode='a').close()


def create_cluster_dirs(root='testdata/net'):
    os.system('rm -rf '+root)
    ensure_dir(root)
                
    for sub in range(8):
        subdir=os.path.join(root, 'sub%d'%(sub+1,))
        ensure_dir(subdir)
        for lse in range(sub*3, (sub+1)*3):
            lsedir=os.path.join(root, 'sub%d/lse%03d'%(sub+1, lse+1))
            ensure_dir(lsedir)
            for d in range(4):
                path=os.path.join(root, 'sub%d/lse%03d/data%d'%(sub+1, lse+1, d+1))
                #print path
                ensure_dir(path)
                pass
            pass
        pass
    pass


def create_dummy_ms(ms_id, partition, num_subbands, root='testdata/net', exclude_lse=['lse019', 'lse020', 'lse021', 'lse011']):
    storage_nodes = sorted(['lse%03d'%(i+1,) for i in range(24) if 'lse%03d'%(i+1,) not in exclude_lse])
    subcluster_names= ['sub'+str(get_subcluster_number(name)) for name in storage_nodes]

    n = int(ceil(num_subbands*1.0/len(storage_nodes)))
    
    msname='L2010_%05d'%(ms_id,)

    for i,lse in enumerate(storage_nodes[:-1]):
        ms_dir=os.path.join(root, subcluster_names[i], lse, partition, msname)
        ensure_dir(ms_dir)
        for sb in range(i*n, (i+1)*n):
            make_empty_file(os.path.join(ms_dir, 'L%05d_SB%03d-uv.MS'%(ms_id, sb)))
            pass
        pass
    ms_dir=os.path.join(root, subcluster_names[-1], storage_nodes[-1], partition, msname)
    ensure_dir(ms_dir)
    for sb in range((len(storage_nodes)-1*n), num_subbands):
        make_empty_file(os.path.join(ms_dir, 'L%05d_SB%03d-uv.MS'%(ms_id, sb)))
        pass
    
    pass




if __name__ == '__main__':
    create_cluster_dirs()
    create_dummy_ms(12345, 'data1', 248)
    create_dummy_ms(20001, 'data1', 248)
    create_dummy_ms(20010, 'data1', 244)
    create_dummy_ms(20020, 'data2', 62)
    create_dummy_ms(20030, 'data3', 248)
    create_dummy_ms(20040, 'data4', 122)
    pass
