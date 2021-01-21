import h5py
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from total_states import states_num, get_scene_names, make_scene_name 

datadir = '../mixed_offline_data/'
data_name = 'images.hdf5'
new_data_name = 'images128.hdf5'
test_scenes = {
        'kitchen':range(21,31),
        'living_room':range(21,31),
        'bedroom':range(21,31),
        'bathroom':range(21,31),
    }

scene_names = get_scene_names(test_scenes)
#185104
pbar = tqdm(total = states_num(test_scenes,preload=data_name))
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
    
    
