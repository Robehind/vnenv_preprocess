"""最短路数据"""
import os
from tqdm import tqdm
import random
import networkx.readwrite as netx
import networkx as nx
import json
from total_states import get_scene_names, make_scene_name
'''输出最短路数据为一个字典，字典有一个特殊的键为map，用于给每个状态编号，其他的键都是物体的实例的字符串，
对应值为一个list，保存所有状态到这个目标的最短路的长度
'''
def write_to_json(ips, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(ips, f)

test_scenes = {
        'kitchen':range(1,21),
        'living_room':range(1,21),
        'bedroom':range(1,21),
        'bathroom':range(1,21),
    }
out_name = 'min_len.json'

graph_name = 'nop_graph.json'
visible_file_name = 'visible_object_map.json'
datadir = '../mixed_offline_data/'

scene_names = get_scene_names(test_scenes)
print(f'making for {len(scene_names)} scenes')

p = tqdm(total = len(scene_names))

for scene_name in scene_names:
    p.update(1)
    
    with open(os.path.join(datadir, scene_name, graph_name),"r",) as f:
        graph_json = json.load(f)
    with open(os.path.join(datadir, scene_name, visible_file_name),"r",) as f:
        visible_data = json.load(f)
    graph = netx.node_link_graph(graph_json).to_directed()
    all_states = list(graph.nodes())
    all_objects_id = visible_data.keys()
    
    save_str = os.path.join(datadir, scene_name, out_name)
    data = {x:[] for x in all_objects_id}
    data['map']={k:all_states.index(k) for k in all_states}
    pd = tqdm(total = len(all_states)*len(all_objects_id))
    for obs_id in all_objects_id:
        all_visible_states = visible_data[obs_id]
        #可视状态可能包含当前动作对应的图中不包含的状态
        all_visible_states = [x for x in all_visible_states if x in all_states]
        for start_state in all_states:

            best_path_len = 9999
            for k in all_visible_states:
                try:
                    path = nx.shortest_path(graph, start_state, k)
                except nx.exception.NetworkXNoPath:
                    print(scene_name)
                    raise Exception
                except nx.NodeNotFound:
                    print(scene_name)
                    raise Exception
                #原地不动的距离为0
                path_len = len(path) - 1
                if path_len < best_path_len:
                    best_path_len = path_len
            data[obs_id].append(best_path_len)
            pd.update(1)

    write_to_json(data, save_str)