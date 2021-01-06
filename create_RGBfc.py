import torchvision.models as models
from torchvision import transforms as T
import torch.nn as nn
import torch
import h5py
import os
from tqdm import tqdm
from total_states import states_num, get_scene_names, make_scene_name 
import random
"""生成RGBpred的fc文件"""
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 7, 4, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2,2),
            #nn.Flatten()
            )

    def forward(self, input_):
        return self.net(input_)

test_scenes = {
        'kitchen':range(16,21),
        'living_room':range(16,21),
        'bedroom':range(16,21),
        'bathroom':range(16,21),
    }
load_model_dir = 'models/enc_092106.dat'

model = Encoder().cuda()
model.load_state_dict(torch.load(load_model_dir))
scene_names = get_scene_names(test_scenes)
print(f'making for {len(scene_names)} scenes')
    
pbar = tqdm(total = states_num(test_scenes, preload='images128.hdf5'))
for n in scene_names:
    datadir = '../mixed_offline_data/'

    fc_writer = h5py.File(os.path.join(datadir,n,'rgbpred_fc_nc.hdf5'), 'w')
    RGBloader = h5py.File(os.path.join(datadir,n,'images128.hdf5'), "r",)
    
    for k in RGBloader.keys():
        pbar.update(1)
        pic = RGBloader[k][:]
        data = T.ToTensor()(pic).unsqueeze(0).cuda()
        out = model(data).detach()
        out = torch.flatten(out)
        fc_writer.create_dataset(k, data = out.cpu().numpy())
    RGBloader.close()
    fc_writer.close()