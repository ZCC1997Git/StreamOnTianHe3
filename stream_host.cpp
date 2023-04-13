#include<iostream>
#include<hthread_host.h>
#include<mpi.h>
#include<float.h>
#include<string>
#include<sys/time.h>
using namespace std;

/*the size of each array*/
#ifndef STREAM_ARRAY_SIZE_PER_THREAD
    #define STREAM_ARRAY_SIZE_PER_THREAD 1024000
#endif

/*the repeated iteration number*/
#ifndef NTIMES
    #define NTIMES 100
#endif

/*the value of offset,used to set aligned access or unaligned access*/
#ifndef OFFSET
    #define OFFSET 0
#endif

/*the type of data*/
#ifndef STREAM_TYPE
    #define STREAM_TYPE double
#endif

/*the number of threads*/
#ifndef NUM_THREADS
    #define NUM_THREADS 24
#endif

#ifndef MIN
#define MIN(x,y) ((x)<(y)?(x):(y))
#endif
#ifndef MAX
#define MAX(x,y) ((x)>(y)?(x):(y))
#endif

#define HLINE "-------------------------------------------------------------"

double	avgtime[4] = {0}, maxtime[4] = {0},mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};
string label[4]={"Copy (1R1W):", "Scale(1R1W):","Add  (2R1W):", "Triad(2R1W):"};
double	bytes[4] = {0};
double	times[4][NTIMES];

void PrintStreamInfo();
void Stream_dsp(int rank);
double mysecond();
void ResultProcess();

int main(int argc,char** argv)
{
    MPI_Init(&argc,&argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    if(rank==0)
        PrintStreamInfo();
    MPI_Barrier(MPI_COMM_WORLD);
    
    Stream_dsp(rank);

    ResultProcess();

    MPI_Finalize();
    return 0;
}

void Stream_dsp(int rank)
{
    int cluster_id=rank;
    /*open the dsp cluster*/
    hthread_dev_open(cluster_id);

    /*load the .dat file,which contains the binary instructions*/
    hthread_dat_load(cluster_id,"stream_kernel.dat");

    unsigned long len_per_thread=(STREAM_ARRAY_SIZE_PER_THREAD+16-1)/16*16;
    unsigned long len_total=len_per_thread*NUM_THREADS;
    unsigned long size=sizeof(STREAM_TYPE)*len_total;
    
    auto a=reinterpret_cast<STREAM_TYPE*>(hthread_malloc(cluster_id,size,HT_MEM_RO));
    auto b=reinterpret_cast<STREAM_TYPE*>(hthread_malloc(cluster_id,size,HT_MEM_RW));
    auto c=reinterpret_cast<STREAM_TYPE*>(hthread_malloc(cluster_id,size,HT_MEM_RW));
    
    /*initialization the array*/
    for(unsigned long j=0; j<len_total; j++) 
    {
	    a[j] = 1.0;
	    b[j] = 2.0;
	    c[j] = 0.0;
	}

    /*create the thread group*/
    auto thread_group=hthread_group_create(cluster_id,NUM_THREADS,NULL,0,0,NULL);

    for(int k=0;k<NTIMES;k++)
    {
        
        /*Copy*/
        {
            unsigned long args[3];
            args[0]=len_per_thread;
            args[1]=(unsigned long)a;
            args[2]=(unsigned long)c;

            times[0][k] = mysecond();
            hthread_group_exec(thread_group,"stream_copy",1,2,args);
            hthread_group_wait(thread_group);
            times[0][k] = mysecond() - times[0][k];
        }

        /*Scale*/
        {
            STREAM_TYPE* scalar = reinterpret_cast<STREAM_TYPE*>(hthread_malloc(cluster_id,sizeof(STREAM_TYPE),HT_MEM_RO));
            scalar[0]=3.0;
            unsigned long args[4];
            args[0]=len_per_thread;
            args[1]=(unsigned long)scalar;
            args[2]=(unsigned long)b;
            args[3]=(unsigned long)c;

            times[1][k] = mysecond();
            hthread_group_exec(thread_group,"stream_scale",1,3,args);
            hthread_group_wait(thread_group);
            times[1][k] = mysecond() - times[1][k];
            hthread_free(scalar);
        }

        /*Add*/
        {
            unsigned long args[4];
            args[0]=len_per_thread;
            args[1]=(unsigned long)a;
            args[2]=(unsigned long)b;
            args[3]=(unsigned long)c;

            times[2][k] = mysecond();
            hthread_group_exec(thread_group,"stream_add",1,3,args);
            hthread_group_wait(thread_group);
            times[2][k] = mysecond() - times[2][k];
        }

        /*Triad*/
        {
            STREAM_TYPE* scalar = reinterpret_cast<STREAM_TYPE*>(hthread_malloc(cluster_id,sizeof(STREAM_TYPE),HT_MEM_RO));
            scalar[0]=3.0;
            unsigned long args[5];
            args[0]=len_per_thread;
            args[1]=(unsigned long)scalar;
            args[2]=(unsigned long)a;
            args[3]=(unsigned long)b;
            args[4]=(unsigned long)c;

            times[3][k] = mysecond();
            hthread_group_exec(thread_group,"stream_triad",1,4,args);
            hthread_group_wait(thread_group);
            times[3][k] = mysecond() - times[3][k];
        }
    }

    /*destroy the thread group*/
    hthread_group_destroy(thread_group);
    /*free the hthread memory*/
    hthread_free(a);
    hthread_free(b);
    hthread_free(c);
    /*close the dsp cluster*/
    hthread_dev_close(cluster_id);
}

/* A gettimeofday routine to give access to the wall
   clock timer on most UNIX-like systems.  */
double mysecond()
{
    struct timeval tp;
    struct timezone tzp;

    gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

int checktick()
{
    #define M 100
    int		i, minDelta, Delta;
    double	t1, t2, timesfound[M];

    /* Collect a sequence of M unique time values from the system.*/
    for (i = 0; i < M; i++) 
    {
        t1 = mysecond();
        while( ((t2=mysecond()) - t1) < 1.0E-6 );
        timesfound[i] = t1 = t2;
	}

    /*
    * Determine the minimum difference between these M values.
    * This result will be our estimate (in microseconds) for the
    * clock granularity.
    */

    minDelta = 1000000;
    for (i = 1; i < M; i++) 
    {
        Delta = (int)( 1.0E6 * (timesfound[i]-timesfound[i-1]));
        minDelta = MIN(minDelta, MAX(Delta,0));
	}
    #undef M

    return(minDelta);
}

void PrintStreamInfo()
{
    int BytesPerWord = sizeof(STREAM_TYPE);

    cout<<endl;
    cout<<HLINE<<endl;
    cout<<"\033[32;01mStream on TianHe (based on STREAM version $Revision: 5.10 $)\033[0m"<<endl;
    cout<<HLINE<<endl;
    cout<<endl;

    int len_per_thread=(STREAM_ARRAY_SIZE_PER_THREAD+16-1)/16*16;
    int len_total=len_per_thread*NUM_THREADS;

    bytes[0]= 2 * sizeof(STREAM_TYPE) * len_total;
    bytes[1]= 2 * sizeof(STREAM_TYPE) * len_total;
    bytes[2]= 3 * sizeof(STREAM_TYPE) * len_total;
    bytes[3]= 3 * sizeof(STREAM_TYPE) * len_total;

    cout<<"Array size = "<<len_per_thread<< "\t Thread Num = "<<NUM_THREADS<<endl;
    cout<<"Memory per array ="<<BytesPerWord * ( len_total/ 1024.0/1024.0)<<" MiB ("<<
    BytesPerWord * ( len_total/ 1024.0/1024.0/1024.0)<<" GiB)"<<endl;

    cout<<"Total memory required ="<<(3.0 * BytesPerWord) * (len_total/ 1024.0/1024.)<<" MiB ("<<
    (3.0 * BytesPerWord) * (len_total/ 1024.0/1024./1024.)<<"GiB)"<<endl;

    cout<<"Each kernel will be executed "<<NTIMES<<"times"<<endl;

    cout<<"\033[32mThe *best* time for each kernel (excluding the first iteration"<<endl; 
    cout<<"will be used to compute the reported bandwidth\033[0m"<<endl;

    cout<<HLINE<<endl;

    int quantum;
    if( (quantum = checktick())>= 1) 
        cout<<"Your clock granularity/precision appears to be "<<quantum<<"microseconds"<<endl;
    else{
        cout<<"Your clock granularity appears to be less than one microsecond"<<endl;
        quantum = 1;
    }
}

void ResultProcess()
{
    /*	--- SUMMARY --- */
    for (int k=1; k<NTIMES; k++) /* note -- skip first iteration */
	{
        for (int j=0; j<4; j++)
        {
            avgtime[j] = avgtime[j] + times[j][k];
            mintime[j] = MIN(mintime[j], times[j][k]);
            maxtime[j] = MAX(maxtime[j], times[j][k]);
        }
	}

    cout<<"Function    Best Rate GB/s  Avg time     Min time     Max time"<<endl;
    for (int j=0; j<4; j++) {
		avgtime[j] = avgtime[j]/(double)(NTIMES-1);
        cout<<label[j]<<'\t'<<1.0E-06 * bytes[j]/1024/mintime[j]<<'\t'<<
            avgtime[j]<<'\t'<<mintime[j]<<'\t'<<maxtime[j]<<endl;
		//printf("%s%12.1f  %11.6f  %11.6f  %11.6f\n", label[j],1.0E-06 * bytes[j]/1024/mintime[j],avgtime[j],mintime[j],maxtime[j]);
    }
}
