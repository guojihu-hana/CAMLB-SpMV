#pragma once
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <string>
#include <omp.h>
#include <immintrin.h>
#include <chrono>

#define FloatType double
// #define FloatType float

int WARM_UP = 100;
int ITERATION = 1000;
int CACHE_LINE_SIZE = 64;

struct Joint
{
    int x;
    int y;
};

void verify_spmv(int r, int *Ap, int *Aj, FloatType *Ax, FloatType *x, FloatType *ry){
    for(int i = 0; i < r; i ++){
        FloatType sum = 0;
        for(int j = Ap[i]; j < Ap[i+1]; j ++){
            sum += x[Aj[j]] * Ax[j];
        }
        ry[i] = sum;
    }
}

inline bool verify(FloatType* y1, FloatType* y2, int v_len)
{
    for (int i = 0; i < v_len; i++) {
        if (fabs(y1[i] - y2[i]) * fabs(y1[i] - y2[i])> 1e-6) {
            std::cout << "Wrong answer i = " << i << " " << y1[i] << " " << y2[i] << std::endl;
            return false;
        }
    }
    return true;
}

template<typename VecT, typename ValT>
inline void initVector(VecT* v, int v_len, ValT init_v)
{   
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_max_threads();
        for (int i = tid * 8; i < v_len; i+=8*num_threads) {
            *(v + i) = init_v;
            *(v + i + 1) = init_v;
            *(v + i + 2) = init_v;
            *(v + i + 3) = init_v;
            *(v + i + 4) = init_v;
            *(v + i + 5) = init_v;
            *(v + i + 6) = init_v;
            *(v + i + 7) = init_v;
        }
    }
}

inline void merge_path_search(const int* Ap, int diagonal, int row_num, int nnz, Joint &point) {
    int begin = std::max(diagonal - nnz, 0);
    int end = std::min(diagonal, row_num);
    while (begin < end) {
        int mid = (begin + end) >> 1 ;
        if (Ap[mid] <= diagonal - mid - 1) {
            begin = mid + 1;
        }
        else {
            end = mid;
        }
    }
    point.x = std::min(row_num, begin);
    point.y = diagonal - begin;
}

inline int binary_search(int* p, int len, int target){
    int b = 0, e = len;
    while(b < e){
        int m = (b + e)/2;
        if(p[m] <= target){
            b = m + 1; 
        }else{
            e = m;
        }
    }
    return std::min(b, len);
}

namespace ReadMatrix {
    using std::string;
    using std::stringstream;
    struct Coordinate {
        int x;
        int y;
        FloatType val;
    };

    inline bool coordcmp(const Coordinate v1, const Coordinate v2)
    {
        if (v1.x != v2.x)
        {
            return (v1.x < v2.x);
        }
        else
        {
            return (v1.y < v2.y);
        }
    }
	inline int removeDuplicates(std::vector<Coordinate> &arr, int len){
        if (len == 0 || len == 1) return len;
        int j = 0;
        for (int i = 0; i < len; i++)
            if (arr[i].x != arr[j].x || arr[i].y != arr[j].y)
                arr[++j] = arr[i];
        return j + 1;
    }
    void readMatrix(string filename, FloatType** val_ptr, int** cols_ptr,
        int** rowDelimiters_ptr, int* n, int* numRows, int* numCols)
    {
        string line;
        string id;
        string object;
        string format;
        string field;
        string symmetry;

        std::ifstream mfs(filename);
        if (!mfs.good())
        {
            std::cerr << "Error: unable to open matrix file " << filename << std::endl;
            exit(1);
        }
        

        int symmetric = 0;
        int pattern = 0;
        int field_complex = 0;

        int nRows;
        int nCols;
        int nElements;
        // read matrix header
        if (getline(mfs, line).eof())
        {
            std::cerr << "Error: file " << filename << " does not store a matrix" << std::endl;
            exit(1);
        }
        stringstream ss;
        ss.clear();
        ss.str(line);
        ss >> id >> object >> format >> field >> symmetry;

        //sscanf(line.c_str(), "%s %s %s %s %s", id, object, format, field, symmetry);

        if (object != "matrix")
        {
            fprintf(stderr, "Error: file %s does not store a matrix\n", filename.c_str());
            exit(1);
        }

        if (format != "coordinate")
        {
            fprintf(stderr, "Error: matrix representation is dense\n");
            exit(1);
        }

        if (field == "pattern")
        {
            pattern = 1;
        }

        if (field == "complex")
        {
            field_complex = 1;
        }

        if (symmetry == "symmetric")
        {
            symmetric = 1;
        }

        while (!getline(mfs, line).eof())
        {
            if (line[0] != '%')
            {
                break;
            }
        }

        ss.clear();
        ss.str(line);
        ss >> nRows >> nCols >> nElements;
        nRows = ((nRows >> 3) + 1) << 3;
        nCols = ((nCols >> 3) + 1) << 3;
        //sscanf(line.c_str(), "%d %d %d", &nRows, &nCols, &nElements);
        int nElements_padding = (nElements % 16 == 0) ? nElements : (nElements + 16) / 16 * 16;
        //int valSize = nElements_padding * sizeof(struct Coordinate);
        int valSize = nElements_padding;
        if (symmetric)
        {
            valSize *= 2;
        }
        std::vector<Coordinate> coords(valSize);
        //coords = (struct Coordinate*)malloc(valSize);
        int index = 0;
        double xx99 = 0;
        while (!getline(mfs, line).eof())
        {
            ss.clear();
            ss.str(line);
            if (pattern)
            {
                ss >> coords[index].x >> coords[index].y;
                coords[index].val = index % 13;
            }
            else if (field_complex)
            {
                ss >> coords[index].x >> coords[index].y >> coords[index].val >> xx99;
            }
            else
            {
                ss >> coords[index].x >> coords[index].y >> coords[index].val;
            }

            // convert into index-0-as-start representation
            coords[index].x--;
            coords[index].y--;    
            index++;
            if (symmetric && coords[index - 1].x != coords[index - 1].y)
            {
                coords[index].x = coords[index - 1].y;
                coords[index].y = coords[index - 1].x;
                coords[index].val = coords[index - 1].val;
                index++;
            }

        }
	    std::sort(coords.begin(), coords.begin() + index, coordcmp);
        int indexBefore = index;
        index = removeDuplicates(coords, index);
        if(indexBefore > index)
            std::cout<<"Remove duplicate:"<<indexBefore - index<<std::endl;
        nElements = index;
        nElements_padding = (nElements % 16 == 0) ? nElements : (nElements + 16) / 16 * 16;

        for (int qq = index; qq < nElements_padding; qq++)
        {
            coords[qq].x = coords[index - 1].x;
            coords[qq].y = coords[index - 1].y;
            coords[qq].val = 0;
        }

        //sort the elements
        std::sort(coords.begin(), coords.begin() + nElements_padding, coordcmp);

        // create CSR data structures
        *n = nElements_padding;
        *numRows = nRows;
        *numCols = nCols;
        *val_ptr = new FloatType[nElements_padding];
        *cols_ptr = new int[nElements_padding];
        *rowDelimiters_ptr = new int[nRows + 2];

        FloatType* val = *val_ptr;
        int* cols = *cols_ptr;
        int* rowDelimiters = *rowDelimiters_ptr;

        rowDelimiters[0] = 0;
        int r = 0;
        int i = 0;
        for (i = 0; i < nElements_padding; i++)
        {
            while (coords[i].x != r)
            {
                rowDelimiters[++r] = i;
            }
            val[i] = coords[i].val;
            cols[i] = coords[i].y;
        }

        for (int k = r + 1; k <= (nRows + 1); k++)
        {
            // rowDelimiters[k] = i - 1;
            rowDelimiters[k] = nElements_padding;
        }
    }
}
