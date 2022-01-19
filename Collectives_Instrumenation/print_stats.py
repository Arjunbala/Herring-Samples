import sys

def ranks_to_nodes(ranks):
    machine_ids = []
    for rank in ranks:
        machine_id = rank / 8
        if machine_id not in machine_ids:
            machine_ids.append(machine_id)
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

commhash_to_ranks_mapping = {}
comm_to_commhash_mapping = {}
comm_to_rank_mapping ={}

# open log file
f = open(sys.argv[1])
lines = f.readlines()

for line in lines:
    if "commHash" in line:
        commHash = line.split(" ")[7].split(",")[0]
        rank = int(line.split(" ")[9])
        if commHash not in commhash_to_ranks_mapping:
            commhash_to_ranks_mapping[commHash] = []
        commhash_to_ranks_mapping[commHash].append(rank)
        comm = line.split(" ")[5].split(",")[0]
        comm_to_commhash_mapping[comm] = commHash
        comm_to_rank_mapping[comm] = rank

per_rank_stats = {}
do_profiling = False

for line in lines:
    if "Starting CUDA profiling" in line:
        do_profiling = True
    if "Stopping CUDA profiling" in line:
        do_profiling = False
    if "sendbuff" in line and do_profiling == True:
        comm = line.split(" ")[20]
        collective = line.split(" ")[4].split(":")[0]
        tensor_size = int(line.split(" ")[12])
        data_type = int(line.split(" ")[14])
        root = int(line.split(" ")[18])
        coll_tuple = (comm_to_commhash_mapping[comm],collective,tensor_size,data_type,root)
        initiating_rank = comm_to_rank_mapping[comm]
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
        commhash,collective,tensor_size,data_type,root = details
        ranks = commhash_to_ranks_mapping[commhash]
        nodes_involved = ranks_to_nodes(ranks)
        across_machines= ""
        if len(nodes_involved) == 1:
            across_machines = "INTRANODE"
        else:
            across_machines = "INTERNODE"
        if collective in ["Reduce","AllGather","AllReduce","Broadcast"]\
        and across_machines in ["INTRANODE","INTERNODE"]:
            print(str(across_machines) + " COLLECTIVE: " + str(collective) + " OCCURRENCES: " + str(count)\
            + " SIZE: "\
            + str(tensor_size) + " TYPE: " + data_type_print(data_type)\
            + " SEND_SIZE: " + str(find_send_size(collective,tensor_size,data_type)) + " bytes"\
            + " RECV_SIZE: " + str(find_recv_size(collective,tensor_size,data_type, len(ranks))) + " bytes"\
            + " RANKS_INVOLVED: " + str(len(ranks)) +" " + str(ranks) + " ROOT: " + str(root))
    print("== ==")
    print("\n")
