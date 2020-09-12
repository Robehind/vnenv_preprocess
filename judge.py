import torch.nn.functional as F
import torch
import h5py
import random
#这个脚本可以用来 简单 判断对于单房间训练而言特征的好坏程度
#会输出整个房间的特征向量的最大值在向量中不同位置的出现次数
score = h5py.File('../mixed_offline_data/FloorPlan27_physics/resnet50_fc_new.hdf5','r')

key = list(score.keys())
print(f'totally {len(key)} fcs')
store = []
fc_len = len(score[key[0]][:].squeeze().tolist())
count = [0 for _ in range(fc_len)]
for k in key:


    a = torch.FloatTensor(score[k][:]).squeeze()

    ss = F.softmax(a.squeeze(),dim=0).numpy().tolist()

    #store.append(ss[818])

    #print(max(a))
    count[torch.argmax(a).item()] +=1
    #for i in ss:
        #if i == 1.:
            #print(ss.index(i))
    #print(a.shape)
    #print(a)
    #print(F.softmax(a.squeeze(),dim=0))

ans = []
for i in count:
    if i is not 0:ans.append(i)

print(ans)
print(f'{max(ans)/len(key)}')