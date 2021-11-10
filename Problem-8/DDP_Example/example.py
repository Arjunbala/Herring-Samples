import os
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10,10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10,5)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def example(rank, world_size):
    print ("In " + str(rank) + " out of " + str(world_size)) 
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # create model and move to appropriate GPU
    net = SimpleModel().to(rank)
    net = DDP(net, device_ids=[rank])
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)

    optimizer.zero_grad()
    output = net(torch.randn(20,10))
    print ("In " + str(rank) + " FW done ")
    labels = torch.randn(20,5).to(rank)
    loss_fn(output, labels).backward()
    print ("In " + str(rank) + " BW done ")
    optimizer.step()
    print ("In " + str(rank) + " optim done ")

    dist.destroy_process_group()

def main():
    world_size = 2
    mp.spawn(example,
            args=(world_size,),
            nprocs=world_size,
            join=True)
if __name__=="__main__":
    main()
