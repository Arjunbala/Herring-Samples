#include<pthread.h>
#include<stdio.h>
#include<unistd.h>

pthread_mutex_t mutex;

void* hello_func(void *p) {
    pthread_mutex_lock(&mutex);
    printf("Hello\n");
    sleep(1);   
    printf("Hello\n");
    pthread_mutex_unlock(&mutex);
    pthread_exit(0);
}

void* world_func(void *p) {
    pthread_mutex_lock(&mutex);
    printf("World\n");
    printf("World\n");
    pthread_mutex_unlock(&mutex);
    pthread_exit(0);
}

int main() {
    pthread_t t1,t2;

    pthread_create(&t1, NULL, hello_func, NULL);
    pthread_create(&t2, NULL, world_func, NULL);

    pthread_join(t2, NULL);
    pthread_join(t1, NULL);
}
