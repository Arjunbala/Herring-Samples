#include <mpi.h>
#include <iostream>
#include <nccl.h>

using namespace std;

__global__ void set_array_value(float *vals, int multiplier) {
    int i = threadIdx.x;
    vals[i] = i*multiplier*1.0;
}

int main() {
    MPI_Init(NULL, NULL);

    int process_rank;
    int world_size;

    // Get rank of process
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    // Get world size
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ncclUniqueId id;
    ncclComm_t comm;
    cudaStream_t stream;

    /*if (process_rank == 0) { // if root generate an ID
        ncclGetUniqueId(&id);
    }
    // broadcast the unique ID to all
    MPI_Bcast((void *) &id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);*/
    
    // now pick a GPU based on rank and initialize NCCL
    cudaSetDevice(process_rank);
    //cudaStreamCreate(&stream);
    //ncclCommInitRank(&comm, world_size, id, process_rank);

    // setup GPU float array and fill some data in it
    float *gpu_tensor;
    int size = 16;
    int ret = cudaMalloc((void **) &gpu_tensor, size * sizeof(float));
    cout<<ret<<endl;
    set_array_value<<<1,size>>>(gpu_tensor, process_rank+1);
    cudaDeviceSynchronize();

    float *cpu_tensor;
    cpu_tensor = (float *) malloc(size * sizeof(float));
    cudaMemcpy(cpu_tensor, gpu_tensor, size * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0;i<size;i++) {
        cout<<cpu_tensor[i]<<" ";
    }
    cout<<endl;
    
    // to store reduce result
    /*float *reduce_result;
    cudaMalloc((void **) &reduce_result, size * sizeof(float));
    
    // now perform all reduce
    ncclAllReduce((const void *) gpu_tensor, (void *) reduce_result, size, 
		    ncclFloat, ncclSum, comm, stream);

    // wait for allreduce to complete
    cudaStreamSynchronize(stream);

    // Verify result by printing on CPU
    float *cpu_result_tensor;
    cpu_result_tensor = (float *) malloc(size * sizeof(float));
    cudaMemcpy(cpu_result_tensor, reduce_result, size * sizeof(float), cudaMemcpyDeviceToHost);
*/
    /*for(int i=0;i<size;i++) {
        cout<<cpu_result_tensor[i]<<" ";
    } 
    cout<<endl;*/

    cudaFree(gpu_tensor);
    //cudaFree(reduce_result);
    //free(cpu_result_tensor);

    MPI_Finalize();
    return 0;
}
