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

    // Finalize the MPI environment.
    MPI_Finalize();
}
