Problem 2: MPI Hello World

1. Write mpi_hello_world.c that initializes MPI, prints its own rank, prints the world size, and uses MPI’s broadcast function to broadcast a value from rank_0 to other ranks. Print the broadcasted value from all ranks. 
2. Build your code with cmake and run it. 

When you are done, something like this must work:

cd project2
mkdir build && cd build
cmake ..
make 
mpirun -np 8 ./mpi_hello_world

References:

1. Initialize MPI (https://github.com/aws/herring/blob/bb6fa88c4025291e4518b2b27ba3774d92eecfbf/commoncpu/MPIInitializer.cpp#L18)
2. Getting rank of process (https://github.com/aws/herring/blob/806b4a74ad28204137e7b5abe324ca6b5392159a/commoncpu/rdmacorrectness.cpp#L297)
3. Broadcasting (https://github.com/aws/herring/blob/fa0ede1417dc2f147a343fdb2c7f3f7ed229f98c/common/HerringClient.cpp#L414)
4. Sample cmake file that links MPI library (https://github.com/aws/herring/blob/master/test/commoncpu/CMakeLists.txt)
