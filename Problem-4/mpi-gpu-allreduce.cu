#include <mpi.h>
#include <iostream>

using namespace std;

__global__ void set_array_value(float *vals, int multiplier) {
    int i = threadIdx.x;
    vals[i] = i * multiplier * 1.0;
}

int main() {
    MPI_Init(NULL, NULL);

    int process_rank;

    // Get rank of process
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    // Assign process to GPU
    cudaSetDevice(process_rank);

    int size = 16;
    float *gpu_tensor;
    cudaMalloc((void**) &gpu_tensor, size * sizeof(float));

    // Use a CUDA Kernel to fill some values in the array
    set_array_value<<<1,size>>>(gpu_tensor,process_rank+1);
    cudaDeviceSynchronize();
    
    float *cpu_tensor;
    cpu_tensor = (float *) malloc(size * sizeof(float));
    cudaMemcpy(cpu_tensor, gpu_tensor, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    float *reduce_result;
    reduce_result = (float *) malloc(size * sizeof(float));
    MPI_Allreduce(cpu_tensor, reduce_result, size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    for(int i=0;i<size;i++) {
        cout<<reduce_result[i]<<" ";
    }
    cout<<endl;

    int ret = cudaMemcpy(gpu_tensor, reduce_result, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaFree(gpu_tensor);
    free(cpu_tensor);
    free(reduce_result); 
     
    MPI_Finalize();

    return 0;
}
