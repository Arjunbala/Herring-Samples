Problem 1: CMake

Herring uses cmake for building libraries and binaries. To get familiar with cmake, 

1. Write a simple .c file that will print “hello world”
2. Write CMakeLists.txt file to build an executable from the .c file. 
3. Build binary from command line and run the binary. 

After you have written the .c file and CMakeLists.txt, something like this must work:

cd project1
mkdir build && cd build 
cmake ..
make
./hello_world

References:

1. A sample cmake file (https://github.com/aws/herring/blob/master/test/commoncpu/CMakeLists.txt)


