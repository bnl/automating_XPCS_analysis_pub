import json
from json import JSONEncoder
import numpy as np


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    
def write_to_file(dic, filename):
    with open(f'{filename}.json', 'w') as openfile:
        json.dump(dic, openfile, cls = NumpyArrayEncoder)

def load_from_file(filename):
    with open(filename, 'r') as openfile:
        x = json.load(openfile)
    return x

def load_result_dict_from_file(filename):

    # first load the dictionary from file
    # the keys here are strings, need to convert them to int

    d = load_from_file(filename)
    new_d = {}
    for key in d.keys():
        new_d[int(key)] = d[key]
    return new_d



