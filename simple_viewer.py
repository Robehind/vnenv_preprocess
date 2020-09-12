import importlib
import os
import json
import h5py
import cv2
import random
dataDir = "../mixed_offline_data/FloorPlan1_physics"
grid_size = 0.25
rot_angle = 45
def step(action, state):
    state = state.split("|")
    z = float(state[1])
    x = float(state[0])
    rotation = int(state[2])
    if action == 119:
        if rotation == 0:
            z += grid_size
        elif rotation == 90:
            x += grid_size
        elif rotation == 180:
            z -= grid_size
        elif rotation == 270:
            x -= grid_size
        elif rotation == 45:
            z += grid_size
            x += grid_size
        elif rotation == 135:
            z -= grid_size
            x += grid_size
        elif rotation == 225:
            z -= grid_size
            x -= grid_size
        elif rotation == 315:
            z += grid_size
            x -= grid_size
        else:
            raise Exception("Unknown Rotation")
    elif action == 100:
        rotation = (rotation + rot_angle) % 360
    elif action == 97:
        rotation = (rotation - rot_angle) % 360
    else:
        print("Unknow Actions")
    return "{:0.2f}|{:0.2f}|{:d}|{:d}".format(x, z, round(rotation), round(0))
#load the graph.json
json_graph_loader = importlib.import_module("networkx.readwrite")
with open(os.path.join(dataDir,"graph.json"), "r") as f:
    graph_json = json.load(f)
graph = json_graph_loader.node_link_graph(graph_json).to_directed()
#load the images.hdf5
images = h5py.File(os.path.join(dataDir,"images.hdf5"), "r")
#fcs = h5py.File(os.path.join(dataDir,"resnet50_fc.hdf5"), "r")
#load the grid.json
with open(os.path.join(dataDir,"grid.json"), "r") as f:
    grid = json.load(f)


lenth = len(grid)
pos = grid[random.choice(range(lenth))]
poskey = "{:0.2f}|{:0.2f}|{:d}|{:d}".format(pos["x"], pos["z"], round(0), round(0))
press_key = 1

while press_key != 27:

    pic = images[poskey][:]
    #print(fcs[poskey][:])
    #RGB to BGR
    pic = pic[:,:,::-1]
    cv2.imshow("showing", pic)
    press_key = cv2.waitKey(0)
    
    next_poskey = step(press_key, poskey)
    #print(graph.neighbors(poskey))
    if next_poskey in graph.neighbors(poskey):
        poskey = next_poskey
    print(poskey)
