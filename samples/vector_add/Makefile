CPU_COMPILER = g++
CUDA_COMPILER = nvcc
HUM_COMPILER = humcc

CUDA_INCLUDES = -I/usr/local/cuda/include
CUDA_LIBRARIES = -L/usr/local/cuda/lib64
HUM_LIBRARIES = -L$(HUMROOT)/lib

CUDA_CPU_CFLAGS = $(CUDA_INCLUDES) $(CUDA_LIBRARIES)
CUDA_CPU_LDFLAGS = -lcudart

HUM_CPU_CFLAGS = $(CUDA_INCLUDES) $(HUM_LIBRARIES)
HUM_CPU_LDFLAGS = -lhum

#CUDA_GPU_CFLAGS = -arch=compute_70 -code=sm_70,compute_70
#HUM_GPU_CFLAGS = -arch=compute_70 -code=sm_70,compute_70

SOURCES = vector_add.cu

CUDA_TARGET = test.cuda
HUM_TARGET = test.hum

all: cuda hum

cuda: $(CUDA_TARGET)

hum: $(HUM_TARGET)

$(CUDA_TARGET): vector_add.cuda.o
	$(CPU_COMPILER) $^ $(CUDA_CPU_CFLAGS) $(CUDA_CPU_LDFLAGS) -o $@

$(HUM_TARGET): vector_add.hum.o
	$(CPU_COMPILER) $^ $(HUM_CPU_CFLAGS) $(HUM_CPU_LDFLAGS) -o $@

vector_add.cuda.o: $(SOURCES)
	$(CUDA_COMPILER) -c $^ $(CUDA_GPU_CFLAGS) -std=c++11 -o $@ 

vector_add.hum.o: $(SOURCES)
	$(HUM_COMPILER) -c $^ $(HUM_GPU_CFLAGS) -std=c++11 -o $@ 

.PHONY: clean run

clean:
	rm -f *.o *.tmp.cu $(CUDA_TARGET) $(HUM_TARGET)

run: all
	./$(TARGET)
