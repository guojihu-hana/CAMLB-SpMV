#include "camlb_spmv.h"

void run_spmv(std::string filename, int mab_type = 1){
    try {
        FloatType *Ax;
        int *Aj, *Ap;
        int nnz, nr, nc;
        ReadMatrix::readMatrix(filename, &Ax, &Aj, &Ap, &nnz, &nr, &nc);
        FloatType *x = (FloatType*)_mm_malloc(nc*sizeof(FloatType), 64);
        FloatType *y = (FloatType*)_mm_malloc(nr*sizeof(FloatType), 64);
        FloatType *ry = (FloatType*)_mm_malloc(nr*sizeof(FloatType), 64);
        initVector(x, nc, 1);
        initVector(y, nr, 0);
        initVector(ry, nr, 0);
        verify_spmv(nr, Ap, Aj, Ax, x, ry);
        spmv_cache_aware(filename, nr, nc, nnz, Aj, Ap, Ax, x, y, false);
        verify(ry,y,nr);
        _mm_free(x);
        _mm_free(y);
        _mm_free(ry);

        delete[] Ax;
        delete[] Aj;
        delete[] Ap;
    }
    catch (std::exception e) {
        std::cout << e.what() << " " << filename << " failed." << std::endl;
    }
}
int main(int argc, char* argv[]) {
    if(argc > 1){
        std::string filename = argv[1];
        run_spmv(filename);
    }
    return EXIT_SUCCESS;
}
