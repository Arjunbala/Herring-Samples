import sys
import argparse

def ranks_to_nodes(ranks):
    machine_ids = []
    for rank in ranks:
        machine_id = rank / 8
        if machine_id not in machine_ids:
            machine_ids.append(machine_id)
    machine_ids.sort()
    return machine_ids

def data_type_to_bytes(datatype):
    mapping = [1,1,1,4,4,4,64,64,16,16,32,32,64,64,16]
    return mapping[datatype]

def data_type_print(datatype):
    mapping = ["ncclInt8", "ncclChar", "ncclUint8",\
    "NcclInt32", "ncclInt", "ncclUint32", "ncclInt64",\
    "ncclUint64", "ncclFloat16", "ncclHalf", "ncclFloat32",\
    "ncclFloat", "ncclFloat64", "ncclDouble", "ncclBFloat16"]
    return mapping[datatype]

def find_send_size(collective, tensor_size, data_type):
    return tensor_size*data_type_to_bytes(data_type)

def find_recv_size(collective, tensor_size, data_type, ranks):
    size  = tensor_size*data_type_to_bytes(data_type)
    if collective == "AllGather":
        size = size * ranks
    return size

parser = argparse.ArgumentParser(description='Print collectives stats')
parser.add_argument('--log_file', help='log file to parse', required=True)
parser.add_argument('--collectives', help='collectives you want to profile separated by comma(,)')
parser.add_argument('--all_collectives', help='profile all possible collectives',action='store_true')
parser.add_argument('--internode', action='store_true')
parser.add_argument('--intranode', action='store_true')
args=parser.parse_args()

collectives_to_profile = []
if args.collectives is not None:
    collectives_to_profile = args.collectives.split(",")
if args.all_collectives is True:
    collectives_to_profile = ["AllReduce","Reduce","ReduceScatter","AllGather","Broadcast"]

collectives_types = []
if args.internode is True:
    collectives_types.append("INTERNODE")
if args.intranode is True:
    collectives_types.append("INTRANODE")

commhash_to_ranks_mapping = {}
comm_and_rank_to_commhash_mapping = {}

# open log file
f = open(args.log_file)
lines = f.readlines()

for line in lines:
    if "commHash" in line:
        commHash = line.split(" ")[7].split(",")[0]
        rank = int(line.split("[")[1].split("]")[0].split(",")[1])
        if commHash not in commhash_to_ranks_mapping:
            commhash_to_ranks_mapping[commHash] = []
        commhash_to_ranks_mapping[commHash].append(rank)
        comm = line.split(" ")[5].split(",")[0]
        comm_and_rank_to_commhash_mapping[(comm,rank)] = commHash

# Sort ranking order to make results more interpretable
for commhash in commhash_to_ranks_mapping:
    commhash_to_ranks_mapping[commhash].sort()

per_rank_stats = {}
do_profiling = False

for line in lines:
    if "Starting CUDA profiling" in line:
        do_profiling = True
    if "Stopping CUDA profiling" in line:
        do_profiling = False
    if "sendbuff" in line and do_profiling == True and len(line.split(" ")) == 24:
        comm = line.split(" ")[20]
        collective = line.split(" ")[4].split(":")[0]
        tensor_size = int(line.split(" ")[12])
        data_type = int(line.split(" ")[14])
        initiating_rank = int(line.split("[")[1].split("]")[0].split(",")[1])
        coll_tuple = (comm_and_rank_to_commhash_mapping[(comm,initiating_rank)],collective,tensor_size,data_type)
        if initiating_rank not in per_rank_stats:
            per_rank_stats[initiating_rank] = {}
        if coll_tuple not in per_rank_stats[initiating_rank]:
            per_rank_stats[initiating_rank][coll_tuple] = 0
        per_rank_stats[initiating_rank][coll_tuple] += 1

for rank in per_rank_stats:
    rank_stats = per_rank_stats[rank]
    print("== Printing stats for rank " + str(rank) + " in decreasing order of occurrence ==")
    inverted_stats= sorted(rank_stats.items(), key=lambda item: item[1], reverse=True)
    for stat in inverted_stats:
        details,count = stat
        commhash,collective,tensor_size,data_type = details
        ranks = commhash_to_ranks_mapping[commhash]
        nodes_involved = ranks_to_nodes(ranks)
        across_machines= ""
        if len(nodes_involved) == 1:
            across_machines = "INTRANODE"
        else:
            across_machines = "INTERNODE"
        if collective in collectives_to_profile\
        and across_machines in collectives_types:
            print(str(across_machines) + " COLLECTIVE: " + str(collective) + " OCCURRENCES: " + str(count)\
            + " SIZE: "\
            + str(tensor_size) + " TYPE: " + data_type_print(data_type)\
            + " SEND_SIZE: " + str(find_send_size(collective,tensor_size,data_type)) + " bytes"\
            + " RECV_SIZE: " + str(find_recv_size(collective,tensor_size,data_type, len(ranks))) + " bytes"\
            + " RANKS_INVOLVED: " + str(len(ranks)) + " " + str(ranks)\
            + " NODES_INVOLVED: " + str(len(nodes_involved)) + " " + str(nodes_involved))
    print("== ==")
    print("\n")
