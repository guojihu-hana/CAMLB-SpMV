#include "spmv_utils.h"
#include "camlb_utils.h"

void spmv_cache_aware_kernel(int numRows, int numCols, int nItems, int* Aj, int* Ap, FloatType* Ax, FloatType* x, FloatType* y, int *Acp, int *thread_task_row_bound, int *thread_task_nnz_bound){
    FloatType value_carry_out[THREAD_NUM];
    int row_carry_out[THREAD_NUM];
    int num_threads = omp_get_max_threads();

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        int i = thread_task_row_bound[tid];
        int end_row = thread_task_row_bound[tid + 1];
        int j = thread_task_nnz_bound[tid];
        int end_nnz = thread_task_nnz_bound[tid + 1];
        for (; i < end_row; i++) {
            FloatType sum = 0.0;
            for (; j < Ap[i + 1]; j++) {
                sum += x[Aj[j]] * Ax[j];
            }
            y[i] = sum;
        }
        FloatType last_partial_sum = 0;
        for (; j < end_nnz; j++) {
            last_partial_sum += x[Aj[j]] * Ax[j];
        }
        value_carry_out[tid] = last_partial_sum;
        row_carry_out[tid] = end_row;
    }
    for(int tid = 0; tid < num_threads; tid++){
        if (row_carry_out[tid] < numRows)
            y[row_carry_out[tid]] += value_carry_out[tid];
    }
}

void spmv_cache_aware(std::string filename, int numRows, int numCols, int nItems, int* Aj, int* Ap, FloatType* Ax, FloatType* x, FloatType* y, bool DEBUG = false) {
    auto preprocessing_start = std::chrono::high_resolution_clock::now();
    int num_threads = omp_get_max_threads();
    int nnz = Ap[numRows];
    int *Acp = (int*)_mm_malloc((nnz + 1) * sizeof(int), 64);
    count_cacheaware_memory_access(numRows, numCols, Ap, Aj, Acp);
    int total_memory_access = Acp[nnz];
    int memory_access_per_thread = (total_memory_access + num_threads - 1 )/ num_threads;
    int *thread_task_row_bound = (int*)_mm_malloc((num_threads + 1) * sizeof(int), 64);
    int *thread_task_nnz_bound = (int*)_mm_malloc((num_threads + 1) * sizeof(int), 64);
    thread_task_row_bound[0] = 0;
    thread_task_nnz_bound[0] = 0;
    #pragma omp parallel for
    for(int tid = 0; tid < num_threads; tid ++){
        int s = std::min(total_memory_access, memory_access_per_thread * tid);
        int e = std::min(total_memory_access, s + memory_access_per_thread);
        int start_pos = binary_search(Acp + 1, nnz, s);
        int end_pos = binary_search(Acp + 1, nnz, e);
        int start_row = binary_search(Ap + 1, numRows, start_pos);
        int end_row = binary_search(Ap + 1, numRows, end_pos);
        thread_task_row_bound[tid + 1] = end_row;
        thread_task_nnz_bound[tid + 1] = end_pos;
    }
    auto preprocessing_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> preprocessing_dur = (preprocessing_end - preprocessing_start);
    auto preprocessing_consume = preprocessing_dur.count() / 1000.0;

    if(DEBUG){
        spmv_cache_aware_kernel(numRows,numCols,nnz,Aj,Ap,Ax,x,y,Acp,thread_task_row_bound,thread_task_nnz_bound);
        return;
    }

    for (int i = 0; i < WARM_UP; i++) {
        spmv_cache_aware_kernel(numRows,numCols,nnz,Aj,Ap,Ax,x,y,Acp,thread_task_row_bound,thread_task_nnz_bound);
    }    
    int	 times = ITERATION;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < times; i++) {
        spmv_cache_aware_kernel(numRows,numCols,nnz,Aj,Ap,Ax,x,y,Acp,thread_task_row_bound,thread_task_nnz_bound);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur = (end - start);
    auto consume = dur.count() / 1000.0;
    double avg_time = consume / times;
    std::cout << "camlb time: " << avg_time << "\tBW: " << 1.0e-9 * nnz * sizeof(FloatType) / (consume / times) << "\tGF: " << 1.0e-9 * 2 * nnz / (consume / times) <<"\tpreprocessing: "<<preprocessing_consume<< std::endl;
}