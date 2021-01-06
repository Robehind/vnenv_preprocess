import h5py
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from total_states import states_num, get_scene_names, make_scene_name 
import random
import numpy as np
from tensorboardX import SummaryWriter

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

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=(2,2),mode='bilinear',align_corners=True),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2,2),mode='bilinear',align_corners=True),
            nn.Conv2d(128, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2,2),mode='bilinear',align_corners=True),
            nn.Conv2d(64, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 4),
            nn.ReLU(inplace=True),
            )
    def forward(self, input_):
        return self.net(input_)

#######################training##################################
datadir = '../mixed_offline_data/'
data_name = 'images128.hdf5'
test_scenes = {
        'kitchen':range(1,16),
        'living_room':range(1,16),
        'bedroom':range(1,16),
        'bathroom':range(1,16),
    }
path_to_save = '.'
print_freq = 10000
save_freq = 1e7
batch_size = 64
total_frames = 1e8

if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)
scene_names = get_scene_names(test_scenes)

enc = Encoder().cuda()
dec = Decoder().cuda()

model = nn.Sequential(
    enc,
    dec,
)

optim = torch.optim.Adam(model.parameters(), lr = 0.0007)

log_writer = SummaryWriter(log_dir = '.')

n_frames = 0
print_gate_frames = print_freq
save_gate_frames = save_freq
loss_record = 0
count = 0


pbar = tqdm(total=total_frames)
while 1:

    random.shuffle(scene_names)
    for s in scene_names:

        loader = h5py.File(os.path.join(datadir,s,data_name),"r",)
        keys = list(loader.keys())
        random.shuffle(keys)
        num = len(keys)
        runs = num // batch_size
        batch_keys = [keys[i*batch_size:(i+1)*batch_size] for i in range(runs)]
        for i in range(runs):
            #####输入128，128，3的255图像时
            data = np.array([loader[x] for x in batch_keys[i]])
            data = torch.tensor(data).to(torch.float32).permute(0,3,1,2).cuda()
            data = data / 255.
            #data = T.ToTensor()(data).cuda()
            out = model(data)
            loss = F.smooth_l1_loss(out, data.detach())
            loss_record += loss.cpu().item()
            count+=1
            loss.backward()
            optim.step()
            model.zero_grad()

            n_frames += batch_size
            pbar.update(batch_size)

            if n_frames >= print_gate_frames:
                print_gate_frames += print_freq
                log_writer.add_scalar("loss", loss_record/count, n_frames)
                loss_record = 0
                count = 0

            if n_frames >= save_gate_frames:
                save_gate_frames += save_freq
                
                enc_to_save = enc.state_dict()
                all_to_save = model.state_dict()
                import time
                start_time = time.time()
                time_str = time.strftime(
                    "%H%M%S", time.localtime(start_time)
                )
                save_path = os.path.join(
                    path_to_save,
                    f"enc_{time_str}.dat"
                )
                torch.save(enc_to_save, save_path)
                save_path = os.path.join(
                    path_to_save,
                    f"model_{time_str}.dat"
                )
                torch.save(all_to_save, save_path)
            
            if n_frames > total_frames:
                loader.close()
                exit()
        loader.close()


        

