#include <assert.h>
#include <stdio.h>
#include <sys/time.h>

int main (int argc, char **argv) {
    int size = (1 << 30)/sizeof(char); // 1GB = 1024*1024*1024 = 2^30

    char *gpu_buffer;
    char *cpu_buffer;

    int unique_char = 0x12;

    // Allocate GPU memory
    cudaMalloc(&gpu_buffer, size * sizeof(char));

    // Set some value for buffer inside GPU memory
    cudaMemset(gpu_buffer, unique_char, size);

    // Now allocate CPU memory
    cpu_buffer = (char *) malloc(size * sizeof(char));

    int num_trials = 100;
    int print_stats_trials = 10;

    int cur_trials = 0;
    unsigned long long total_time_usecs = 0;

    while(cur_trials <= num_trials) {
	struct timeval begin,end;
	gettimeofday(&begin, NULL);
        // transfer data
	cudaMemcpy(cpu_buffer, gpu_buffer, size * sizeof(char), cudaMemcpyDeviceToHost);
        gettimeofday(&end, NULL);
        // measuring timing
	total_time_usecs += (end.tv_sec- begin.tv_sec)*1000000 + (end.tv_usec- begin.tv_usec);
	// validate data transfered
	for (int i = 0; i < size; i++) {
            assert(cpu_buffer[i] == (char) unique_char);
	}
	// increment trails
        cur_trials++;
	// print stats if required
	if(cur_trials % print_stats_trials == 0) {
	    printf("Average Throughput = %f Gbps\n", (cur_trials*8.0)/(total_time_usecs/(1000000)));   
	}	
    }
}
