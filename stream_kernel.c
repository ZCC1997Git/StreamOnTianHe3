#include<compiler/m3000.h>
#include<hthread_device.h>
#include<assert.h>

/*the type of data*/
#ifndef STREAM_TYPE
    #define STREAM_TYPE double
#endif

/*the number of threads*/
#ifndef NUM_THREADS
    #define NUM_THREADS 24
#endif

/*the chunksize on AM*/
#ifndef CHUNK_SIZE
    #define CHUNK_SIZE 1024
#endif

__global__ void stream_copy(unsigned long len_per_thread,STREAM_TYPE* a,STREAM_TYPE* c)
{
    int tid=get_thread_id();
    lvector STREAM_TYPE* a_am=(lvector STREAM_TYPE*)vector_malloc(CHUNK_SIZE*sizeof(STREAM_TYPE));
    lvector STREAM_TYPE* c_am=(lvector STREAM_TYPE*)vector_malloc(CHUNK_SIZE*sizeof(STREAM_TYPE));

    STREAM_TYPE* a_local=&(a[tid*len_per_thread]);
    STREAM_TYPE* c_local=&(c[tid*len_per_thread]);

    int num_chunk=len_per_thread/CHUNK_SIZE;
    assert(len_per_thread%CHUNK_SIZE==0);
    
    if(tid<NUM_THREADS)
    {
        for(long chunk=0;chunk<num_chunk;chunk++,a_local+=CHUNK_SIZE,c_local+=CHUNK_SIZE)
        {
            vector_load(a_local,a_am,CHUNK_SIZE*sizeof(STREAM_TYPE));

            for(int i=0;i<CHUNK_SIZE/16;i++)
                c_am[i]=a_am[i];
            
            vector_store(c_am,c_local,CHUNK_SIZE*sizeof(STREAM_TYPE));
        }
    }

    vector_free(a_am);
    vector_free(c_am);
}

__global__ void stream_scale(unsigned long len_per_thread,STREAM_TYPE* scaler_v, STREAM_TYPE* b,STREAM_TYPE* c)
{
    int tid=get_thread_id();
    lvector STREAM_TYPE* b_am=(lvector STREAM_TYPE*)vector_malloc(CHUNK_SIZE*sizeof(STREAM_TYPE));
    lvector STREAM_TYPE* c_am=(lvector STREAM_TYPE*)vector_malloc(CHUNK_SIZE*sizeof(STREAM_TYPE));

    STREAM_TYPE* b_local=&(b[tid*len_per_thread]);
    STREAM_TYPE* c_local=&(c[tid*len_per_thread]);

    int num_chunk=len_per_thread/CHUNK_SIZE;
    assert(len_per_thread%CHUNK_SIZE==0);
    
    STREAM_TYPE scaler_t=scaler_v[0];

    if(tid<NUM_THREADS)
    {
        for(long chunk=0;chunk<num_chunk;chunk++,b_local+=CHUNK_SIZE,c_local+=CHUNK_SIZE)
        {
            vector_load(c_local,c_am,CHUNK_SIZE*sizeof(STREAM_TYPE));

            for(int i=0;i<CHUNK_SIZE/16;i++)
                b_am[i]=scaler_t*c_am[i];
            
            vector_store(b_am,b_local,CHUNK_SIZE*sizeof(STREAM_TYPE));
        }
    }

    vector_free(b_am);
    vector_free(c_am);
}


__global__ void stream_add(unsigned long len_per_thread,STREAM_TYPE* a, STREAM_TYPE* b,STREAM_TYPE* c)
{
    int tid=get_thread_id();
    lvector STREAM_TYPE* a_am=(lvector STREAM_TYPE*)vector_malloc(CHUNK_SIZE*sizeof(STREAM_TYPE));
    lvector STREAM_TYPE* b_am=(lvector STREAM_TYPE*)vector_malloc(CHUNK_SIZE*sizeof(STREAM_TYPE));
    lvector STREAM_TYPE* c_am=(lvector STREAM_TYPE*)vector_malloc(CHUNK_SIZE*sizeof(STREAM_TYPE));

    STREAM_TYPE* a_local=&(a[tid*len_per_thread]);
    STREAM_TYPE* b_local=&(b[tid*len_per_thread]);
    STREAM_TYPE* c_local=&(c[tid*len_per_thread]);

    int num_chunk=len_per_thread/CHUNK_SIZE;
    assert(len_per_thread%CHUNK_SIZE==0);
    
    if(tid<NUM_THREADS)
    {
        for(long chunk=0;chunk<num_chunk;chunk++,b_local+=CHUNK_SIZE,c_local+=CHUNK_SIZE)
        {
            vector_load(a_local,a_am,CHUNK_SIZE*sizeof(STREAM_TYPE));
            vector_load(b_local,b_am,CHUNK_SIZE*sizeof(STREAM_TYPE));

            for(int i=0;i<CHUNK_SIZE/16;i++)
                c_am[i]=a_am[i]+b_am[i];
            
            vector_store(c_am,c_local,CHUNK_SIZE*sizeof(STREAM_TYPE));
        }
    }

    vector_free(a_am);
    vector_free(b_am);
    vector_free(c_am);
}

__global__ void stream_triad(unsigned long len_per_thread,STREAM_TYPE* scaler_v, STREAM_TYPE* a,STREAM_TYPE* b,STREAM_TYPE* c)
{
    int tid=get_thread_id();

    lvector STREAM_TYPE* a_am=(lvector STREAM_TYPE*)vector_malloc(CHUNK_SIZE*sizeof(STREAM_TYPE));
    lvector STREAM_TYPE* b_am=(lvector STREAM_TYPE*)vector_malloc(CHUNK_SIZE*sizeof(STREAM_TYPE));
    lvector STREAM_TYPE* c_am=(lvector STREAM_TYPE*)vector_malloc(CHUNK_SIZE*sizeof(STREAM_TYPE));

    STREAM_TYPE* a_local=&(a[tid*len_per_thread]);
    STREAM_TYPE* b_local=&(b[tid*len_per_thread]);
    STREAM_TYPE* c_local=&(c[tid*len_per_thread]);

    int num_chunk=len_per_thread/CHUNK_SIZE;
    assert(len_per_thread%CHUNK_SIZE==0);
    
    STREAM_TYPE scaler_t=scaler_v[0];

    if(tid<NUM_THREADS)
    {
        for(long chunk=0;chunk<num_chunk;chunk++,b_local+=CHUNK_SIZE,c_local+=CHUNK_SIZE)
        {
            vector_load(b_local,b_am,CHUNK_SIZE*sizeof(STREAM_TYPE));
            vector_load(c_local,c_am,CHUNK_SIZE*sizeof(STREAM_TYPE));
            for(int i=0;i<CHUNK_SIZE/16;i++)
                a_am[i]=b_am[i]+scaler_t*c_am[i];
            
            vector_store(a_am,a_local,CHUNK_SIZE*sizeof(STREAM_TYPE));
        }
    }

    vector_free(a_am);
    vector_free(b_am);
    vector_free(c_am);
}