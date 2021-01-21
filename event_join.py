from tensorboard.backend.event_processing import event_accumulator
from tensorboardX import SummaryWriter
import os

datadir = '/home/zhiyu/EXPS/rlfrom0'
outdir = './join'
ea = []
events_path = []
for root, dirs, files in os.walk(datadir):
    for f in files:
        if f.split('.')[0] == 'events':
            events_path.append(os.path.join(root, f))
events_path = sorted(events_path,key=lambda a:os.path.getmtime(a))
for f in events_path:
    ea.append(event_accumulator.EventAccumulator(f))
log_writer = SummaryWriter(log_dir=outdir)
last_step = {}
last_frame = 0
for event in ea:
    event.Reload()
    for k in event.scalars.Keys():
        data1 = event.scalars.Items(k)
        if k not in last_step.keys():
            last_step.update({k:0})
        b = 0
        if k == 'n_frames':
            b = last_frame
        for d in data1:
            log_writer.add_scalar(k, d.value + b, d.step+last_step[k])
        if k == 'n_frames':
            last_frame += data1[-1].value
        last_step[k] += data1[-1].step
        
log_writer.close()

