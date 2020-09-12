import json
import os
datadir = '../mixed_offline_data/'
scene_name = 'FloorPlan2_physics'
with open(os.path.join(datadir,scene_name, "visible_object_map.json"),"r",) as f:
    metadata = json.load(f)
aa = [x.split('|')[0] for x in metadata]
aa = set(aa)
aa = list(aa)
print(aa)