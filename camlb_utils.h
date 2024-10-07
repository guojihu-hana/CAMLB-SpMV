#include<vector>
#include<queue>
#define THREAD_NUM 96
inline void count_cacheaware_memory_access(int numRows, int numCols, int *Ap, int *Aj, int *Acp_cache_aware, int CACHE_SIZE=72*1024*1024){
    int memory_access = 0;
    
    int last_x_cache_addr = -1;
    int last_y_cache_addr = -1;
    int last_Ax_cache_addr = -1;
    int last_Aj_cache_addr = -1;
    int last_Ap_cache_addr = -1;
    int MAX_CACHELINE_NUM = CACHE_SIZE / CACHE_LINE_SIZE;
    int MAX_INT_IN_CACHELINE = CACHE_LINE_SIZE / sizeof(int);
    int MAX_FLOATTYPE_IN_CACHELINE = CACHE_LINE_SIZE / sizeof(FloatType);
    std::vector<int8_t> bs(numCols / MAX_FLOATTYPE_IN_CACHELINE + 1, 0);
    std::queue<int> sliding_window; 
    int max_window_size = MAX_CACHELINE_NUM / (THREAD_NUM); // Total 96 threads in experiments.
    Acp_cache_aware[0] = 0;
    for(int i = 0, last_Ap_cache_addr = -1, last_y_cache_addr = -1; i < numRows; i++){
        if((i / MAX_INT_IN_CACHELINE) != last_Ap_cache_addr){
            last_Ap_cache_addr = i / MAX_INT_IN_CACHELINE;
            memory_access++;
        }
        if(((i + 1) / MAX_INT_IN_CACHELINE) != last_Ap_cache_addr){
            last_Ap_cache_addr = (i + 1) / MAX_INT_IN_CACHELINE;
            memory_access++;
        }
        //last_x_cache_addr = -1;
        for(int j = Ap[i]; j < Ap[i + 1]; j++){
            int idx = Aj[j] / MAX_FLOATTYPE_IN_CACHELINE;
            if(last_x_cache_addr != idx){
                last_x_cache_addr = idx;
                if(!bs[last_x_cache_addr]){
                    if(sliding_window.size() >= max_window_size){
                        bs[sliding_window.front()] = 0;
                        sliding_window.pop();   
                    }
                    memory_access++;
                    bs[last_x_cache_addr] = 1;
                    sliding_window.push(last_x_cache_addr);
                }
            }
            if(last_Ax_cache_addr != j / MAX_FLOATTYPE_IN_CACHELINE){
                last_Ax_cache_addr = j / MAX_FLOATTYPE_IN_CACHELINE;
                memory_access++;
            }
            if(last_Aj_cache_addr != j / MAX_INT_IN_CACHELINE){
                last_Aj_cache_addr = j / MAX_INT_IN_CACHELINE;
                memory_access++;
            }
            Acp_cache_aware[j + 1] = memory_access;
        }
        if((i / MAX_FLOATTYPE_IN_CACHELINE) != last_y_cache_addr){
            last_y_cache_addr = i / MAX_FLOATTYPE_IN_CACHELINE;
            memory_access++;
        }
    }
}