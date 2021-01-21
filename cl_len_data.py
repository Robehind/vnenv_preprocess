"""用于不断增加距离的CL的数据"""
import os
from tqdm import tqdm
import random
import networkx.readwrite as netx
import networkx as nx
import json
from total_states import get_scene_names, make_scene_name

def write_to_json(ips, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(ips, f)

test_scenes = {
        'kitchen':range(1,21),
        'living_room':range(1,21),
        'bedroom':range(1,21),
        'bathroom':range(1,21),
    }
min_len_name = 'min_len.json'
out_name = 'cl_len.json'
visible_file_name = 'visible_object_map.json'
datadir = '../mixed_offline_data/'

scene_names = get_scene_names(test_scenes)
print(f'making for {len(scene_names)} scenes')

p = tqdm(total = len(scene_names))

for scene_name in scene_names:
    
    p.update(1)
    
    with open(os.path.join(datadir, scene_name, min_len_name),"r",) as f:
        min_len = json.load(f)
    save_str = os.path.join(datadir, scene_name, out_name)
    map_ = min_len.pop('map')
    inv_map = ['' for _ in range(len(map_.keys()))]
    for k in map_.keys():
        inv_map[map_[k]]=k
    insts_list = list(min_len.keys())
    objs_list = list(set([x.split('|')[0] for x in insts_list]))
    data = {'map':inv_map}
    for obj in objs_list:
        insts = [k for k in insts_list if k.split("|")[0] == obj]
        max_len = min([max(min_len[i]) for i in insts])
        data[obj] = [[] for _ in range(max_len+1)]
        for k in map_.keys():
            idx = min([min_len[i][map_[k]] for i in insts])
            data[obj][idx].append(map_[k])

    write_to_json(data, save_str)