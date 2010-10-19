import os
import ma

def is_list(obj):
    return type(obj) == type([])

def is_masked_array(obj):
    return type(obj) == type(ma.array([]))

def full_path_listdir(directory):
    return map(lambda d: directory+d, os.listdir(directory))

