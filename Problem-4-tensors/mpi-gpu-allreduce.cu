#include <torch/torch.h>
#include <iostream>
#include <mpi.h>

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

    // Create a tensor on GPU
    int size = 16;
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, process_rank);
    torch::Tensor gpu_tensor = torch::full(size, process_rank, options);
   
    // copy to CPU
    torch::Tensor cpu_tensor = gpu_tensor.cpu();

    // now do an allreduce
    int *source_buffer = cpu_tensor.data_ptr<int>();
    int *result_buffer = (int *) malloc(size * sizeof(int));
    MPI_Allreduce(source_buffer, result_buffer, size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    for(int i=0;i<size;i++) {
        cout<<result_buffer[i]<<" ";
    }
    cout<<endl;
    options = torch::TensorOptions().dtype(torch::kInt32);
    gpu_tensor = torch::from_blob(result_buffer, size, options).cuda();
    cout<<gpu_tensor<<endl;

    MPI_Finalize();
    return 0;
}
