
# include <stdio.h>
# include <unistd.h>
# include <math.h>
# include <float.h>
# include <limits.h>
# include <sys/time.h>
# include <omp.h>

#ifndef STREAM_ARRAY_SIZE
#   define STREAM_ARRAY_SIZE	10000
#endif

#ifndef NTIMES
#   define NTIMES	10
#endif

#ifndef OFFSET
#   define OFFSET	0
#endif

# define HLINE "-------------------------------------------------------------\n"

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

#ifndef STREAM_TYPE
#define STREAM_TYPE double
#endif

static STREAM_TYPE	a[STREAM_ARRAY_SIZE+OFFSET],b[STREAM_ARRAY_SIZE+OFFSET],c[STREAM_ARRAY_SIZE+OFFSET];

static double avgtime[5] = {0}, maxtime[5] = {0}, mintime[5] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};

static char	*label[5] = {"Copy (1R1W):", "Scale(1R1W):","Add  (2R1W):", "Triad(2R1W):","Read (3R0W):"};

static double	bytes[5] = {
    2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE
    };

double mysecond();
void checkSTREAMresults();
int checktick();

int main()
{
    int			quantum;
    int			BytesPerWord;
    int			k;
    ssize_t		j;
    STREAM_TYPE		scalar;
    double		t, times[5][NTIMES];
    volatile STREAM_TYPE total=0;
    /* --- SETUP --- determine precision and check timing --- */

    printf(HLINE);
    printf("STREAM version $Revision: 5.10 $\n");
    printf(HLINE);
    BytesPerWord = sizeof(STREAM_TYPE);
    printf("This system uses %d bytes per array element.\n",
	BytesPerWord);

    printf(HLINE);


    printf("Array size = %llu (elements), Offset = %d (elements)\n" , (unsigned long long) STREAM_ARRAY_SIZE, OFFSET);
    printf("Memory per array = %.1f MiB (= %.1f GiB).\n", 
	BytesPerWord * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.0),
	BytesPerWord * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.0/1024.0));
    printf("Total memory required = %.1f MiB (= %.1f GiB).\n",
	(3.0 * BytesPerWord) * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.),
	(3.0 * BytesPerWord) * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024./1024.));
    printf("Each kernel will be executed %d times.\n", NTIMES);
    printf(" The *best* time for each kernel (excluding the first iteration)\n"); 
    printf(" will be used to compute the reported bandwidth.\n");


    printf(HLINE);
    #pragma omp parallel 
    {
        #pragma omp master
        {
            k = omp_get_num_threads();
            printf ("Number of Threads requested = %i\n",k);
        }
    }


    k = 0;
    #pragma omp parallel
    #pragma omp atomic 
            k++;
    printf ("Number of Threads counted = %i\n",k);


    /* Get initial value for system clock. */
    #pragma omp parallel for
    for (j=0; j<STREAM_ARRAY_SIZE; j++) {
	    a[j] = 1.0;
	    b[j] = 2.0;
	    c[j] = 0.0;
	}

    printf(HLINE);

    if  ( (quantum = checktick()) >= 1) 
        printf("Your clock granularity/precision appears to be "
            "%d microseconds.\n", quantum);
    else {
        printf("Your clock granularity appears to be "
            "less than one microsecond.\n");
        quantum = 1;
    }

    t = mysecond();
    #pragma omp parallel for
    for (j = 0; j < STREAM_ARRAY_SIZE; j++)
		a[j] = 2.0E0 * a[j];
    t = 1.0E6 * (mysecond() - t);

    printf("Each test below will take on the order"
	" of %d microseconds.\n", (int) t  );
    printf("   (= %d clock ticks)\n", (int) (t/quantum) );
    printf("Increase the size of the arrays if this shows that\n");
    printf("you are not getting at least 20 clock ticks per test.\n");

    printf(HLINE);

    printf("WARNING -- The above is only a rough guideline.\n");
    printf("For best results, please be sure you know the\n");
    printf("precision of your system timer.\n");
    printf(HLINE);
    
    /*	--- MAIN LOOP --- repeat test cases NTIMES times --- */
    scalar = 3.0;
    for (k=0; k<NTIMES; k++)
	{
        /*COPY*/
        times[0][k] = mysecond();
        #pragma omp parallel for
            for (j=0; j<STREAM_ARRAY_SIZE; j++)
                c[j] = a[j];
        times[0][k] = mysecond() - times[0][k];

        /*Scale*/
        times[1][k] = mysecond();
        #pragma omp parallel for
        for (j=0; j<STREAM_ARRAY_SIZE; j++)
            b[j] = scalar*c[j];
        times[1][k] = mysecond() - times[1][k];

        /*Add*/
        times[2][k] = mysecond();
        #pragma omp parallel for
            for (j=0; j<STREAM_ARRAY_SIZE; j++)
                c[j] = a[j]+b[j];
        times[2][k] = mysecond() - times[2][k];

        /*Triad*/
        times[3][k] = mysecond();
        #pragma omp parallel for
        for (j=0; j<STREAM_ARRAY_SIZE; j++)
            a[j] = b[j]+scalar*c[j];
        times[3][k] = mysecond() - times[3][k];

         /*Triad*/
        register STREAM_TYPE t=0;
        times[4][k] = mysecond();
        #pragma omp parallel reduction(+:t)
        {
            register STREAM_TYPE tt=t;
            #pragma omp for
            for (j=0; j<STREAM_ARRAY_SIZE; j++)
                tt+=a[j]+b[j]+c[j];
            t+=tt;
        }
        times[4][k] = mysecond() - times[4][k];
        total+=t;
    }

    /*	--- SUMMARY --- */
    for (k=1; k<NTIMES; k++) /* note -- skip first iteration */
	{
        for (j=0; j<5; j++)
        {
            avgtime[j] = avgtime[j] + times[j][k];
            mintime[j] = MIN(mintime[j], times[j][k]);
            maxtime[j] = MAX(maxtime[j], times[j][k]);
        }
	}
    
    printf("Function    Best Rate GB/s  Avg time     Min time     Max time\n");
    for (j=0; j<5; j++) {
		avgtime[j] = avgtime[j]/(double)(NTIMES-1);
		printf("%s%12.1f  %11.6f  %11.6f  %11.6f\n", label[j],1.0E-06 * bytes[j]/1024/mintime[j],avgtime[j],mintime[j],maxtime[j]);
    }
    printf(HLINE);

    /* --- Check Results --- */
    checkSTREAMresults();
    printf(HLINE);

    return 0;
}


int checktick()
{
    # define M	20
    int		i, minDelta, Delta;
    double	t1, t2, timesfound[M];

    /*Collect a sequence of M unique time values from the system. */

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


/* A gettimeofday routine to give access to the wall
   clock timer on most UNIX-like systems.  */

#include <sys/time.h>

double mysecond()
{
        // struct timeval tp;
        // struct timezone tzp;
        // int i;

        // i = gettimeofday(&tp,&tzp);
        // return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
        return omp_get_wtime();
}

#ifndef abs
    #define abs(a) ((a) >= 0 ? (a) : -(a))
#endif
void checkSTREAMresults ()
{
	STREAM_TYPE aj,bj,cj,scalar;
	STREAM_TYPE aSumErr,bSumErr,cSumErr;
	STREAM_TYPE aAvgErr,bAvgErr,cAvgErr;
	double epsilon;
	ssize_t	j;
	int	k,ierr,err;

    /* reproduce initialization */
	aj = 1.0;
	bj = 2.0;
	cj = 0.0;
    /* a[] is modified during timing check */
	aj = 2.0E0 * aj;
    /* now execute timing loop */
	scalar = 3.0;
	for (k=0; k<NTIMES; k++)
    {
        cj = aj;
        bj = scalar*cj;
        cj = aj+bj;
        aj = bj+scalar*cj;
    }

    /* accumulate deltas between observed and expected results */
	aSumErr = 0.0;
	bSumErr = 0.0;
	cSumErr = 0.0;
	for (j=0; j<STREAM_ARRAY_SIZE; j++) {
		aSumErr += abs(a[j] - aj);
		bSumErr += abs(b[j] - bj);
		cSumErr += abs(c[j] - cj);
	}
	aAvgErr = aSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;
	bAvgErr = bSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;
	cAvgErr = cSumErr / (STREAM_TYPE) STREAM_ARRAY_SIZE;

	if (sizeof(STREAM_TYPE) == 4) {
		epsilon = 1.e-6;
	}
	else if (sizeof(STREAM_TYPE) == 8) {
		epsilon = 1.e-13;
	}
	else {
		printf("WEIRD: sizeof(STREAM_TYPE) = %lu\n",sizeof(STREAM_TYPE));
		epsilon = 1.e-6;
	}

	err = 0;
	if (abs(aAvgErr/aj) > epsilon) {
		err++;
		printf ("Failed Validation on array a[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
		printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",aj,aAvgErr,abs(aAvgErr)/aj);
		ierr = 0;
		for (j=0; j<STREAM_ARRAY_SIZE; j++) {
			if (abs(a[j]/aj-1.0) > epsilon) {
				ierr++;
			}
		}
		printf("     For array a[], %d errors were found.\n",ierr);
	}

	if (abs(bAvgErr/bj) > epsilon) {
		err++;
		printf ("Failed Validation on array b[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
		printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",bj,bAvgErr,abs(bAvgErr)/bj);
		printf ("     AvgRelAbsErr > Epsilon (%e)\n",epsilon);
		ierr = 0;
		for (j=0; j<STREAM_ARRAY_SIZE; j++) {
			if (abs(b[j]/bj-1.0) > epsilon) {
				ierr++;
			}
		}
		printf("     For array b[], %d errors were found.\n",ierr);
	}

	if (abs(cAvgErr/cj) > epsilon) {
		err++;
		printf ("Failed Validation on array c[], AvgRelAbsErr > epsilon (%e)\n",epsilon);
		printf ("     Expected Value: %e, AvgAbsErr: %e, AvgRelAbsErr: %e\n",cj,cAvgErr,abs(cAvgErr)/cj);
		printf ("     AvgRelAbsErr > Epsilon (%e)\n",epsilon);
		ierr = 0;
		for (j=0; j<STREAM_ARRAY_SIZE; j++) {
			if (abs(c[j]/cj-1.0) > epsilon) {
				ierr++;
			}
		}
		printf("     For array c[], %d errors were found.\n",ierr);
	}

	if (err == 0) {
		printf ("Solution Validates: avg error less than %e on all three arrays\n",epsilon);
	}
}


