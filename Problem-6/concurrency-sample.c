#include<pthread.h>
#include<stdio.h>
#include<unistd.h>

void* hello_func(void *p) {
    printf("Hello\n");
    sleep(1);   
    printf("Hello\n");
    pthread_exit(0);
}

void* world_func(void *p) {
    printf("World\n");
    printf("World\n");
    pthread_exit(0);
}

int main() {
    pthread_t t1,t2;

    pthread_create(&t1, NULL, hello_func, NULL);
    pthread_create(&t2, NULL, world_func, NULL);

    pthread_join(t2, NULL);
    pthread_join(t1, NULL);
}
