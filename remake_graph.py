import os
from tqdm import tqdm
import random
import networkx.readwrite as netx
import json
import networkx as nx
"""通过trans.json重制一个基于nx的图，用于算最短路"""
action_dict = {
        'MoveAhead':['m0'],
        'TurnLeft':['r-45'],
        'TurnRight':['r45'],
    }
new_graphdata_name = 'nop_graph.json'
rotate_angle = 45
horizons = [0]
rotations = [x*rotate_angle for x in range(0,360//rotate_angle)]
move_list = [0, 1, 1, 1, 0, -1, -1, -1]
move_list = [x*0.25 for x in move_list]

def write_to_json(ips, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(ips, f)

def str_split(state):
    x, z, rot, hor = state.split('|')
    return float(x),float(z),int(rot),int(hor)

def get_str(x, z, r, h):
    return "{:0.2f}|{:0.2f}|{:d}|{:d}".format(x,z,r,h)

def action_interpret(act_str, source, trans_data):
    s_x, s_z, s_r, s_h = str_split(source)
    success = True
    for str_ in act_str:
        angle = int(str_[1:])
        if str_[0] == 'm':
            abs_angle = (angle + s_r + 360) % 360
            if trans_data[get_str(s_x, s_z, abs_angle, s_h)]:
                s_x += move_list[(abs_angle//45)%8] 
                s_z += move_list[(abs_angle//45+2)%8]
            else:
                success = False
        elif str_[0] == 'r':
            s_r = (angle + s_r + 360) % 360
        elif str_[0] == 'p':
            s_h = (angle + s_h + 360) % 360
        if not success:
            return None, False
    return get_str(s_x, s_z, s_r, s_h), True

def get_scene_names(train_scenes):
    
    return [
        make_scene_name(k, i) for k in train_scenes.keys() for i in train_scenes[k]
    ]

def make_scene_name(scene_type, num):
    mapping = {"kitchen":'', "living_room":'2', "bedroom":'3', "bathroom":'4'}
    front = mapping[scene_type]
    endd = '_physics' if (front == '' or front == '2') else ''
    if num >= 10 or front == '':
        return "FloorPlan" + front + str(num) + endd
    return "FloorPlan" + front + "0" + str(num) + endd

def remake_graph(scene_name, data_name):

    G = nx.DiGraph() 
    datadir = '../mixed_offline_data/'
    with open(
            os.path.join(datadir, scene_name, 'trans.json'),"r",
        ) as f:
            trans_data = json.load(f)
    all_states = list(trans_data.keys())
    save_str = os.path.join(datadir, scene_name, data_name)

    for k in all_states:
        _, _, rot, hor = str_split(k)
        
        if rot not in rotations or hor not in horizons:
            continue
        for act_str in action_dict.values():
            state, success = action_interpret(act_str, k, trans_data)
            if success:
                G.add_edge(k, state)

    data = nx.node_link_data(G)
    #data['directed']=True
    write_to_json(data, save_str)

test_scenes = {
        'kitchen':range(1,21),
        'living_room':range(1,21),
        'bedroom':range(1,21),
        'bathroom':range(1,21),
    }

scene_names = get_scene_names(test_scenes)
print(f'making for {len(scene_names)} scenes')

p = tqdm(total = len(scene_names))
#258154
for n in scene_names:
    p.update(1)
    remake_graph(n, new_graphdata_name)