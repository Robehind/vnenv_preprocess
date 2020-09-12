import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import torch
import h5py
import os
from tqdm import tqdm
import random
"""生成新的fc和fc score文件"""
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

def make_fc_score(scene_name):
    #print("making ",scene_name)
    datadir = '../mixed_offline_data/'

    fc_writer = h5py.File(os.path.join(datadir,scene_name,'resnet50_fc_new.hdf5'), 'w')
    score_writer = h5py.File(os.path.join(datadir,scene_name,'resnet50_score.hdf5'), 'w')
    RGBloader = h5py.File(os.path.join(datadir,scene_name,'images.hdf5'),"r",)
    
    for k in RGBloader.keys():
        pbar.update(1)
        x = RGBloader[k][:]
        x = transforms.ToTensor()(x).unsqueeze(0)
        x = x.cuda()

        resnet_fc = resnet50_fc(x).squeeze()
        resnet_s = resnet50_s(resnet_fc).squeeze()
        #print(resnet_score.shape)
        fc_writer.create_dataset(k, data = resnet_fc.cpu().numpy())
        score_writer.create_dataset(k, data = resnet_s.cpu().numpy())
        #print(resnet_fc.shape)
        #break
    RGBloader.close()
    fc_writer.close()
    score_writer.close()
    #print(resnet_score)

resnet50 = models.resnet50(pretrained=True)
#resnet50 = resnet50.cuda()
for p in resnet50.parameters():
    p.requires_grad = False
resnet50.eval()

resnet50_fc = list(resnet50.children())[:-1]
resnet50_fc = nn.Sequential(*resnet50_fc)
resnet50_fc = resnet50_fc.cuda()
for p in resnet50_fc.parameters():
    p.requires_grad = False
resnet50_fc.eval()

resnet50_s = list(resnet50.children())[-1:]
resnet50_s = nn.Sequential(*resnet50_s)
resnet50_s = resnet50_s.cuda()
for p in resnet50_s.parameters():
    p.requires_grad = False
resnet50_s.eval()

test_scenes = {
        'kitchen':range(16,31),
        'living_room':range(16,31),
        'bedroom':range(16,31),
        'bathroom':range(16,31),
    }

scene_names = get_scene_names(test_scenes)
print(f'making for {len(scene_names)} scenes')

count=0
for s in scene_names:
    RGBloader = h5py.File(os.path.join('../mixed_offline_data/',s,'images.hdf5'),"r",)
    num = len(list(RGBloader.keys()))
    #print(num)
    count += num
    RGBloader.close()
    
pbar = tqdm(total = count)
for n in scene_names:
    make_fc_score(n)