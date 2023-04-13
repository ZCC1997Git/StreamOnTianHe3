ENV_ROOT = ${PROG_ENV}/hthreads
DSP_ROOT=${PROG_ENV}/dsp_compiler

CFLAGS=--std=c11 -Wall -c -O3 -fenable-m3000 -ffunction-sections -flax-vector-conversions -I./ -I$(ENV_ROOT)/include -I$(DSP_ROOT)/include/
LDFLAGS= -L$(ENV_ROOT)/lib --gc-sections -Tdsp.lds


ALL:stream_dsp Stream Stream_MT

stream_dsp: stream_host.cpp  stream_kernel.dat
	mpicxx -O3 -march=native -std=c++17  -I $(ENV_ROOT)/include  -fopenmp  -Wall stream_host.cpp  -o stream_dsp $(ENV_ROOT)/lib/libhthread_host.a -lpthread -fopenmp

stream_kernel.dat: stream_kernel.out
	$(DSP_ROOT)/bin/MT-3000-makedat -J stream_kernel.out
	
stream_kernel.out: stream_kernel.o
	$(DSP_ROOT)/bin/MT-3000-ld $(LDFLAGS) stream_kernel.o $(ENV_ROOT)/lib/libhthread_device.a $(DSP_ROOT)/lib/vlib3000.a $(DSP_ROOT)/lib/slib3000.a -o stream_kernel.out

stream_kernel.o: stream_kernel.c
	$(DSP_ROOT)/bin/MT-3000-gcc $(CFLAGS) stream_kernel.c -o stream_kernel.o

Stream:stream.c
	gcc stream.c -DSTREAM_ARRAY_SIZE=200000000 -DNTIMES=20 -DOFFSET=0  -O3  -mtune=native -march=native -fopenmp -DTUNED -o Stream

Stream_MT:Stream_MT.c
	gcc Stream_MT.c -DSTREAM_ARRAY_SIZE=200000000 -DNTIMES=20 -DOFFSET=0 -DSTREAM_TYPE=double -O3  -mtune=native -march=native -fopenmp -DTUNED -o Stream_MT

.PHONY: clean
clean:
	rm Stream stream_dsp  stream_kernel.dat stream_kernel.out stream_kernel.o Stream_MT
