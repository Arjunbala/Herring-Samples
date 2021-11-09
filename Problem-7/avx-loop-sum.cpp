#include <immintrin.h>
#include <stdio.h>

int main() {
    int array1[24],array2[24];
    // Fill in some data
    for(int i=0;i<24;i++) {
        array1[i] = array2[i] = i+1;
    }

    int result[24];
    for(int i=0;i<24;i+=8) {
        __m256i vector1 = _mm256_loadu_si256((__m256i *) &array1[i]);
	__m256i vector2 = _mm256_loadu_si256((__m256i *) &array2[i]);
	__m256i avx_res = _mm256_add_epi32(vector1, vector2);
        _mm256_storeu_si256((__m256i *) &result[i], avx_res);
    }

    // print the result
    for(int i=0;i<24;i++) {
        printf("%d ", result[i]);
    }
}
