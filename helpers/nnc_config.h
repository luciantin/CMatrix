#ifndef CMATRIX_NNC_CONFIG_H
#define CMATRIX_NNC_CONFIG_H

#define DEBUG 1
#define PROFILE 0
#define SANITY_CHECKS 0

#define NNC_PRINT_TOP_LEFT_MATRIX_START       1
#define NNC_PRINT_BOTTOM_LEFT_MATRIX_START    0
#define NNC_PRINT_LAYER_PATH 0

#define NNC_RAND_ALGO NNC_RAND_ALGO_STDLIB_SRAND
#define NNC_MIN_RAND -1
#define NNC_MAX_RAND 1

#define NNC_ENABLE_FILE_RW 1

#define NNC_MATRIX_MULTIPLY_SQUARE_ALGO NNC_MATRIX_MULTIPLY_SQUARE_ALGO_ITERATIVE

#define NNC_CALCULATE_EXECUTION_TIME 1

#define NNC_PARALLEL 0
#define NNC_PARALLEL_MAX_PARALLEL_JOBS 19
#if NNC_PARALLEL == 1
#include <pthread.h>
#endif

enum NNC_RAND_ALGO_TYPE {
    NNC_RAND_ALGO_STDLIB_SRAND,
    NNC_RAND_ALGO_LINEAR_CONGRUENTIAL_GENERATOR_ANSIC,
};

enum NNC_MATRIX_MULTIPLY_SQUARE_ALGO_TYPE {
    NNC_MATRIX_MULTIPLY_SQUARE_ALGO_ITERATIVE,
    NNC_MATRIX_MULTIPLY_SQUARE_ALGO_STRASSEN,
    NNC_MATRIX_MULTIPLY_SQUARE_ALGO_WINOGRAD,
    NNC_MATRIX_MULTIPLY_SQUARE_ALGO_KARSTADT_SCHWARTZ
};

//enum NNC_MATRIX_MULTIPLY_NON_SQUARE_ALGO {
//
//};

#define nnc_mtype           float
#define nnc_vector          nnc_mtype*
#define nnc_uint            unsigned int
#define nnc_matrix          nnc_mtype**
#define nnc_end_of_string   '\0'
#define nnc_null            ((void *)0)
#define nnc_bool            int
#define nnc_true            1
#define nnc_false           0

nnc_mtype NNCGetRandomMType();
nnc_mtype NNCGetRandomUnsignedMType();


#if DEBUG
    #include <stdio.h>
    #define dprintf printf
    #define dputs puts
#else
    #define dprintf(...) /**/
    #define dputs(...) /**/
#endif

#if NNC_ENABLE_FILE_RW == 1
    #include <stdio.h>
#endif


#if PROFILE
//    #include <stdio.h>
//    #define dprintf printf
//    #define dputs puts
#else
//    #define dprintf(...) /**/
//    #define dputs(...) /**/
#endif


#if NNC_RAND_ALGO == NNC_RAND_ALGO_STDLIB_SRAND
    #include <stdlib.h>
    #define nnc_init_rand srand;
    #define nnc_get_rand_int rand()
    #define nnc_rand_max RAND_MAX
#elif NNC_RAND_ALGO == NNC_RAND_ALGO_LINEAR_CONGRUENTIAL_GENERATOR_ANSIC
    static unsigned int _nnc_rand_seed = 9898;
    static int _nnc_get_rand_int(){
        _nnc_rand_seed = _nnc_rand_seed * 1103515245 + 12345;
        return((unsigned)(_nnc_rand_seed/65536) % 32768);
    }
    #define nnc_init_rand _nnc_rand_seed =
    #define nnc_get_rand_int _nnc_get_rand_int()
    #define nnc_rand_max 65536
#endif

#if NNC_CALCULATE_EXECUTION_TIME == 1
    #include <time.h>
    #define time_type long double
    #define dgetTime()  ((long double)clock() / (long double)CLOCKS_PER_SEC)
    #define dgetTimeDiff(begin, end) ((long double)((end - begin)))
#elif
    #define dgetTime() (long double)0
    #define dgetTimeDiffDouble(start, end) (long double)0
#endif

#endif //CMATRIX_NNC_CONFIG_H
