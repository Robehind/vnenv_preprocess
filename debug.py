
import json
import os
from tqdm import tqdm
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
        'kitchen':[29]#range(1,31),
        #'living_room':range(1,31),
        #'bedroom':range(1,31),
        #'bathroom':range(1,31),
    }

scene_names = get_scene_names(test_scenes)
datadir = '../mixed_offline_data/'
#1.25|1.25|270|0 and 0.25|-0.50|270|0
#w1 = '1.25|1.25|0|0'
#    w2 =  '-1.00|-0.50|0|0'
#1.00|2.00|90|0 and -1.25|1.00|0|0
#pbar = tqdm(total = len(scene_names))
import networkx.readwrite as netx
import networkx as nx
for scene_name in scene_names:
    with open(os.path.join(datadir, scene_name, 'threeACT_graph.json'),"r",) as f:
        graph_json = json.load(f)
    graph = netx.node_link_graph(graph_json).to_directed()
    print(scene_name)
    w1 = '-0.25|-0.50|90|0'
    w2 =  '0.50|-0.75|270|0'
    all_states = list(graph.nodes())
    if w1 in all_states and w2 in all_states:
        print('get path')
        path = nx.shortest_path(graph, w1, w2)
    # with open(os.path.join(datadir,scene_name, "visible_object_map.json"),"r",) as f:
    #     metadata = json.load(f)
    # aa = [x.split('|')[0] for x in metadata]
    # aa = set(aa)
    # #pbar.update(1)
    # print(scene_name)
    # print(list(train_targets.intersection(aa)))
    # #print(aa)