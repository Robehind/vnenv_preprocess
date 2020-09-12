import h5py
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


datadir = '../mixed_offline_data/'

test_scenes = {
        'kitchen':range(1,31),
        'living_room':range(1,31),
        'bedroom':range(1,31),
        'bathroom':range(1,31),
    }

scene_names = get_scene_names(test_scenes)

count=0
pbar = tqdm(total = len(scene_names))
for s in scene_names:

    RGBloader = h5py.File(os.path.join(datadir,s,'images.hdf5'),"r",)
    num = len(list(RGBloader.keys()))
    #print(num)
    count += num
    pbar.update(1)
    RGBloader.close()
print('#################')
print(count)
