#include <mpi.h>
#include <stdio.h>

int main (int argc, char **argv){

    // Init MPI envirovment
    MPI_Init(NULL, NULL);

    int process_rank;
    int world_size;

    // Get rank of process
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    // Get world size
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    printf("Process Rank = %d :: World Size = %d\n", process_rank, world_size);

    int broadcast_int;
    if(process_rank == 0) { // root
        broadcast_int = 711;
    } else {
	broadcast_int = 0;
    }

    // Broadcast the value to all processes
    MPI_Bcast(&broadcast_int, 1 /* count */, MPI_INT, 0 /* root */, MPI_COMM_WORLD);

    // Now print out received value in all processes
    printf("Process Rank = %d :: Broadcast received = %d\n", process_rank, broadcast_int);

    // Finalize the MPI environment.
    MPI_Finalize();
}
