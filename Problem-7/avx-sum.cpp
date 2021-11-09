#include <immintrin.h>
#include <stdio.h>

__m256i add(__m256i a, __m256i b) {
    return _mm256_add_epi32(a,b);
}

int main(){
    __m256i array1 = _mm256_set_epi32(1, 1, 1, 1, 1, 1, 1, 1);
    __m256i array2 = _mm256_set_epi32(1, 1, 1, 1, 1, 1, 1, 1);
    __m256i array3 = _mm256_set_epi32(1, 1, 1, 1, 1, 1, 1, 1);
    __m256i array4 = _mm256_set_epi32(1, 1, 1, 1, 1, 1, 1, 1);
    __m256i array5 = _mm256_set_epi32(1, 1, 1, 1, 1, 1, 1, 1);

    __m256i result = add(array1, array2);
    result = add(result, array3);
    result = add(result, array4);
    result = add(result, array5);

    int *result_display = (int*) &result;
    for(int i=0;i<8;i++) {
        printf("%d ", result_display[i]);
    }
}
