import os
import shutil

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

test_scenes = {
        'kitchen':range(1,31),
        'living_room':range(1,31),
        'bedroom':range(1,31),
        'bathroom':range(1,31),
    }

scene_names = get_scene_names(test_scenes)
file_to_copy = ['trans.json']
s_datadir = './mixed_offline_data/'
d_datadir = './data/'

for s in scene_names:
    os.mkdir(os.path.join(d_datadir,s))
    for f in file_to_copy:
        s_path = os.path.join(s_datadir,s,f)
        d_path = os.path.join(d_datadir,s,f)
        shutil.copy(s_path, d_path)

