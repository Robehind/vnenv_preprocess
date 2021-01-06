import h5py
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F

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
data_name = 'images.hdf5'
new_data_name = 'images128.hdf5'
test_scenes = {
        'kitchen':range(1,21),
        'living_room':range(1,21),
        'bedroom':range(1,21),
        'bathroom':range(1,21),
    }

scene_names = get_scene_names(test_scenes)
#185104
pbar = tqdm(total = 185104)
for s in scene_names:
    data = {}

    loader = h5py.File(os.path.join(datadir,s,data_name),"r",)
    num = len(list(loader.keys()))
    for k in loader.keys():
        vobs = torch.tensor(loader[k][:]).unsqueeze(0)
        vobs = vobs.permute(0,3,1,2)
        vobs = F.interpolate(vobs,(128,128))
        vobs = vobs.permute(0,2,3,1).squeeze(0)

        data[k] = vobs.numpy()
        #data[k] = loader[k][:].squeeze()
    #print(num)
    
    loader.close()
    writer = h5py.File(os.path.join(datadir,s,new_data_name),"w",)
    for k in data:
        writer.create_dataset(k, data = data[k])
        pbar.update(1)
    writer.close()
    
    
