
      #include <iostream>
      #include <nccl.h>
      int main()
      {
        std::cout << NCCL_MAJOR << '.' << NCCL_MINOR << '.' << NCCL_PATCH << std::endl;
        int x;
        ncclGetVersion(&x);
        return x == NCCL_VERSION_CODE;
      }
