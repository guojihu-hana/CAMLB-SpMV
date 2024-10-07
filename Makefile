CC = icpx
#CFLAGS = -O3 -qmkl -fopenmp -mavx512vl -mavx512dq -mavx512f
CFLAGS = -O3 -qmkl -fopenmp -xCORE-AVX512

SOURCES = camlb_spmv.cpp
TARGETS = $(SOURCES:.cpp=)

all: $(TARGETS)

%: %.cpp
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)