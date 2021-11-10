python -m torch.distributed.launch --nnode=1 --node_rank=0 --nproc_per_node=8 example.py --local_world_size=8
