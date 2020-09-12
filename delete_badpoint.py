import json
import os
from tqdm import tqdm
def is_close(g, state):
    nei = g.neighbors(state)
    for t in nei:
        x, z, _, _ = state.split('|')
        x1, z1, _, _ = t.split('|')
        if x != x1 and z != z1:
            return False
    return True

def write_to_json(ips, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(ips, f)
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

train_targets = set([
            "Toaster", "Microwave", "Fridge",
            "CoffeeMaker", "GarbageCan", "Box", "Bowl",
            
            "Pillow", "Laptop", "Television",
            "GarbageCan", "Box", "Bowl",
            "HousePlant", "Lamp", "Book", "AlarmClock",
        "Sink", "ToiletPaper", "SoapBottle", "LightSwitch"])
test_scenes = {
        'kitchen':range(21,31),
        'living_room':range(21,31),
        'bedroom':range(21,31),
        'bathroom':range(21,31),
    }

scene_names = get_scene_names(test_scenes)
datadir = '../mixed_offline_data/'
graph_name = 'threeACT_graph.json'
#1.25|1.25|270|0 and 0.25|-0.50|270|0
#w1 = '1.25|1.25|0|0'
#    w2 =  '-1.00|-0.50|0|0'
pbar = tqdm(total = len(scene_names))
import networkx.readwrite as netx
import networkx as nx
for scene_name in scene_names:
    save_str = os.path.join(datadir, scene_name, graph_name)
    pbar.update(1)
    with open(os.path.join(datadir, scene_name, graph_name),"r",) as f:
        graph_json = json.load(f)
    graph = netx.node_link_graph(graph_json).to_directed()
    all_states = set(list(graph.nodes()))
    #print(len(all_states))
    
    ss = max(nx.strongly_connected_components(graph))
    bad_set = list(all_states.difference(ss))
    #print([] == bad_set)
    if bad_set != []:
        print(bad_set)
        print(f'bad point detected in {scene_name}')
    graph.remove_nodes_from(bad_set)
            

    data = nx.node_link_data(graph)
    #data['directed']=True
    write_to_json(data, save_str)
