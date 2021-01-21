import h5py
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from total_states import states_num, get_scene_names, make_scene_name 
#300x300:[0.5265, 0.4560, 0.3682]), 'std': tensor([0.0540, 0.0554, 0.0567]
#128x128:[0.5269, 0.4565, 0.3687]), 'std': tensor([0.0540, 0.0554, 0.0567]
def get_mean_std(
    scenes, 
    data_name = 'images.hdf5',
    datadir = '../mixed_offline_data/',
    ):
    
    scene_names = get_scene_names(scenes)

    trans = T.Compose([
        T.ToTensor(),
        #T.Normalize(mean=[0,0,0],std=[1,1,1],inplace=True)
    ])
    means = []
    stds = []
    pbar = tqdm(total = states_num(scenes, preload=data_name))
    for s in scene_names:
        data = {}
        loader = h5py.File(os.path.join(datadir,s,data_name),"r",)
        num = len(list(loader.keys()))
        for k in loader.keys():
            vobs = trans(loader[k][:])
            #vobs = torch.tensor(loader[k][:])
            means.append(torch.mean(vobs, dim=[1,2]))
            stds.append(torch.std(vobs, dim=[1,2]))
            pbar.update(1)
        loader.close()

    mean = torch.mean(torch.stack(means), dim=0)
    std = torch.std(torch.stack(stds), dim=0)
    return dict(mean=mean,std=std)
    
if __name__ == "__main__":
    test_scenes = {
            'kitchen':range(1,31),
            'living_room':range(1,31),
            'bedroom':range(1,31),
            'bathroom':range(1,31),
        }
    print(get_mean_std(test_scenes, data_name='images128.hdf5'))