#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <pthread.h>
#include <time.h>
//gcc fft_final.c -o fft_final -lpthread 
typedef double complex cplx;
// Define PI as a constant
#define PI 3.14159265358979323846
long int N = 8;
long num_threads = 8;
long num_iters = 10;
cplx in_x[8];

long threadcounter = 0;
// Define a complex number type
pthread_mutex_t mutexA;
//cplx x[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64};
//cplx x[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
// Swap two complex numbers
long NUM_ELEMENTS = 8;

pthread_barrier_t barrier1;
struct timespec start, end;

void swap(cplx *a, cplx *b) {
  cplx temp = *a;
  *a = *b;
  *b = temp;
}

// Reverse the bits of an integer
unsigned long reverse(unsigned long x, long n) {
  unsigned long result = 0;
  long i;
  for (i = 0; i < n; i++) {
    result = (result << 1) | (x & 1);
    x >>= 1;
  }
  return result;
}

// Define a struct to hold the inputs to fft_helper_multi
typedef struct {
  long n;    // Length of the vector
      // Logarithm base 2 of the subproblem size
  cplx *x;  // Pointer to the input vector
  long end;
  long start;
  long num_threads;
  long totalrun;
  long thread_id;
  long jprotect;
} fft_helper_multi_input;

typedef struct {
  long s;    // Logarithm base 2 of the subproblem size
  cplx *x;  // Pointer to the input vector
  long n;
  long m;
} fft_helper_input;

typedef struct {
  cplx *x;  // Pointer to the input vector
  long k; //loop iteration
  long m; //2^s computation
  long j; // j loop iterator
  cplx w; // iterator for w
  cplx w_m; // w_m multiplier
  long n;//size of vector

} fft_helper_bad_input;

typedef struct {
  cplx *x;  // Pointer to the input vector
  long m; //2^s computation
  long kstart;
  long kend;
} fft_helper_kthreaded_input;

void swapV2(double *a, double *b) {
    double temp = *a;
    *a = *b;
    *b = temp;
}
void bitReversal(int N, double *x, double *y)
{
    int n = N / 2;
    // Bit-reverse the input array
    int i, j = 0, k;
    for (i = 0; i < N-1; i++) {
        if (i < j) {
            swapV2(&x[i], &x[j]);
            swapV2(&y[i], &y[j]);
        }
        k = n;
        while (k <= j) {
            j -= k;
            k /= 2;
        }
        j += k;
    }
}

typedef struct {
  int N;    // Length of the vector
  double *x;  // Pointer to the input vector
  double *y;  // Pointer to the input vector
} fft_worker_input;

void *fft_worker(void *arg){
    fft_worker_input *input = (fft_worker_input *) arg;
    int N = input->N;
    // Perform the FFT
    int n = N / 2;
    int i, j, k;
    double c, s, t1, t2;
    for (k = 1; k < N; k *= 2) {
        for (i = 0; i < N; i += 2*k) {
            for (j = i; j < i+k && j+k < N; j++) {
                c = cos(-PI*j/k);
                s = sin(-PI*j/k);
                t1 = c*input->x[j+k] - s*input->y[j+k];
                t2 = s*input->x[j+k] + c*input->y[j+k];
                    input->x[j+k] = input->x[j] - t1;
                    input->y[j+k] = input->y[j] - t2;
                    input->x[j] += t1;
                    input->y[j] += t2;
            }
        }
    }
}
// Function to perform the FFT
void fft_serialV2(int N, double *x, double *y) {
    bitReversal(N, x, y);

    fft_worker_input input_arg;
        input_arg.N = N;
        input_arg.x = x;
        input_arg.y = y;
     fft_worker(&input_arg);
}

void *fft_helper_kthreaded(void *arg)
{
    fft_helper_kthreaded_input *input = (fft_helper_kthreaded_input *) arg;
    cplx *x = input->x;
    long m = input->m;
    long kstart = input->kstart;
    long kend = input->kend;
    long mDiv2 = m/2;
    cplx w_m = cexp(-2.0 * PI * I / m);
    // printf("kstart: %d, kend: %d\n", kstart, kend);
    for (long k = kstart; k < kend; k += m) {
        // The initial twiddle factor
        cplx w = 1.0;
        //For each pair of elements in the subproblem
        for (long j = 0; j <mDiv2; j++) {
            // The indices of the elements
            long t = k + j;
            long u = t + m / 2;

            // The butterfly operation
            cplx temp = w * x[u];
            x[u] =  x[t] - temp;
            x[t] += temp;
            w = w * w_m;
        }
    }
        // printf("\nTime: %f\n", interval(start,end));
}

// Perform an iterative FFT on a vector of complex numbers
void fft_k_thready(cplx *x,long n, long num_threads) {
  // Assume n is a power of 2
  long logn = log2(n);

  // Rearrange the elements of x according to the bit-reversed order
  for (long i = 0; i < n; i++) {
    unsigned long j = reverse(i, logn);
    if (j > i) {
      swap(&x[i], &x[j]);
    }
  }

  
  // Perform the butterfly operations
  for (long s = 1; s <= logn; s++) {
    long m = 1 << s;
    // The twiddle factor
    // For each subproblem
    fft_helper_kthreaded_input input[num_threads];
    pthread_t tid[num_threads];
    long k_per_thread = ceil(1.0*n/num_threads);
    if (k_per_thread < m){
        k_per_thread = m;
    }
    long next_kstart = 0;
    for(long th_count = 0; th_count < num_threads; th_count++){
        input[th_count].m = m;
        input[th_count].x = x;
        input[th_count].kstart = next_kstart;
        next_kstart += k_per_thread;
        next_kstart = ceil(1.0*next_kstart/m)*m;
        input[th_count].kend = k_per_thread + input[th_count].kstart;
        if(input[th_count].kend > n)
        {
            input[th_count].kend = n;
        }
        if(input[th_count].kstart > n)
        {
            break;
        }
        fft_helper_kthreaded(&input[th_count]);
        // pthread_create(&tid[th_count], NULL, fft_helper_kthreaded, &input[th_count]);
    }
    // for(long th_count = 0; th_count < num_threads; th_count++){
    //     pthread_join(tid[th_count], NULL);
    // }
  }
}

// Print a vector of complex numbers
void print_vector(cplx *x, long n) {
  printf("[");
  for (long i = 0; i < n; i++) {
    printf("%.2f%+.2fi", creal(x[i]), cimag(x[i]));
    if (i < n - 1) {
      printf(", ");
    }
  }
}

double interval(struct timespec start, struct timespec end)
{
  struct timespec temp;
  temp.tv_sec = end.tv_sec - start.tv_sec;
  temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  if (temp.tv_nsec < 0) {
    temp.tv_sec = temp.tv_sec - 1;
    temp.tv_nsec = temp.tv_nsec + 1000000000;
  }
  return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
}

void *fft_helper_bad(void *arg)
{
    fft_helper_bad_input *input = (fft_helper_bad_input *) arg;
    long k = input -> k;
    long m = input -> m;
    long j = input -> j;
    long n = input->n;

    // The indices of the elements
    long t = k + j;
    long u = t + m / 2;

    // The butterfly operation
    cplx temp = input -> w * input -> x[u];
    input -> x[u] = input -> x[t] - temp;
    input -> x[t] = input -> x[t] + temp;
    //printf("k: %d, j: %d, t: %d, u: %d\n", k, j, t, u);
}

// Perform an iterative FFT on a vector of complex numbers
void fft_bad(cplx *x, long n, long num_threads) {
  // Assume n is a power of 2
  long logn = log2(n);

  // Rearrange the elements of x according to the bit-reversed order
  for (long i = 0; i < n; i++) {
    unsigned long j = reverse(i, logn);
    if (j > i) {
      swap(&x[i], &x[j]);
    }
  }

  
  // Perform the butterfly operations
  for (long s = 1; s <= logn; s++) {
    long m = 1 << s;
    // The twiddle factor
    cplx w_m = cexp(-2.0 * PI * I / m);
    // For each subproblem
    for (long k = 0; k < n; k += m) {
      // The initial twiddle factor
        cplx w = 1.0;
      // For each pair of elements in the subproblem
        long mDiv2 = m/2;
        pthread_t tid[mDiv2];
        fft_helper_bad_input input[mDiv2];
        for (long j = 0; j <mDiv2; j++) {
            //fft_helper_bad(&input);
            input[j].x = x;
            input[j].j = j;
            input[j].k = k;
            input[j].m = m;
            input[j].n = n;
            input[j].w_m = w_m;
            input[j].w = w;
            w = input[j].w * input[j].w_m;
            if(j >= num_threads){
                pthread_join(tid[j-num_threads], NULL);
            }
            pthread_create(&tid[j], NULL, fft_helper_bad, &input[j]);
            //pthread_join(tid[j], NULL);
        }
        for(long j = mDiv2-num_threads; j <mDiv2; j++){
            pthread_join(tid[j], NULL);
        }
    }
    //printf("\n");
  }
}

void *fft_helper(void *arg)
{
    fft_helper_input *input = (fft_helper_input *) arg;
    cplx *x = input->x;
    long m = input->m;
    long n = input->n;
    long mDiv2 = m/2;
    cplx w_m = cexp(-2.0 * PI * I / m);
    // printf("kstart: %d, kend: %d\n", kstart, kend);
    for (long k = 0; k < n; k += m) {
        // The initial twiddle factor
        cplx w = 1.0;
        //For each pair of elements in the subproblem
        for (long j = 0; j <mDiv2; j++) {
            // The indices of the elements
            long t = k + j;
            long u = t + m / 2;

            // The butterfly operation
            cplx temp = w * x[u];
            x[u] =  x[t] - temp;
            x[t] += temp;
            w = w * w_m;
        }
    }
}

// Perform an iterative FFT on a vector of complex numbers
void fft(cplx *x, long n) {
  // Assume n is a power of 2
  long logn = log2(n);

  // Rearrange the elements of x according to the bit-reversed order
  //for (long i = 0; i < n; i++) {
  //  unsigned long j = reverse(i, logn);
  //  if (j > i) {
  //    swap(&x[i], &x[j]);
  //  }
  //}
  
  // Perform the butterfly operations
  for (long s = 1; s <= logn; s++) {
    // Create a struct to hold the inputs to fft_helper
    fft_helper_input input = {s, x, n, pow(2,s)};
        fft_helper(&input);
  }
}

void *fft_helper_multi(void *arg)
{
  //pthread_mutex_lock(&mutexA);
    fft_helper_multi_input *input = (fft_helper_multi_input *) arg;
  long logn = log2(input->n);
  long rc;
  for (long s = 1; s <= logn ; s++) {  
    long m = 1 << s;
    // The twiddle factor
    cplx w_m = cexp(-2.0 * PI * I / m);
    // Starting point for each thread
    long kset;
    long jset;
    long wset;
    long TimesBefore;
    //Determines the starting k, j, and twiddle value for each thread based on the thread id and s value
    if(m/2 > (input->totalrun/2) * input->thread_id){
    kset = 0;
    jset = input->thread_id*(input->totalrun/2);
    wset = jset;
    }
    else if(input->thread_id == 0){
      kset = 0;
      jset = 0;
      wset = 0;
    }
    else{
      kset = ((input->thread_id*(input->totalrun/2))/(m/2)) * m;
      TimesBefore = (((input->thread_id)*(input->totalrun/2)));
      jset = TimesBefore % (m/2); 
      wset = jset;
    }
    
    for (long k = kset; k < input->totalrun+kset; k += m) {
      // The initial twiddle factor
      cplx w = 1.0;
      //If a thread is picking up a subproblem that is not the first one, it needs to update the twiddle factor to match where it wouldve started
      if(wset != 0){
        for(long i = 0; i < wset; i++){
          w = w * w_m;
        }
      }
      // For each pair of elements in the subproblem
      for (long j = jset; j < m / 2 && j < (jset+input->totalrun/2); j++) {
        // Calculates the index of each of the 2 elements
        long t = k + j;
        long u = t + (m / 2);
        // The butterfly operation
        cplx temp = w * input->x[u];
        input->x[u] = input->x[t] - temp;
        input->x[t] = input->x[t] + temp;
        // Update the twiddle factor
        w = w * w_m;
      }
    }
    //This barrier prevents the threads from moving on to the next s value until all threads have finished the current s value
    rc = pthread_barrier_wait(&barrier1);
        if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) {
          printf("Could not wait on barrier (return code %d)\n", rc);
           exit(-1);
          }
  }
    // Return NULL to indicate success
    return NULL;
}

void fft_parallel(cplx *x, long n, long num_threads) {
    // Assume n is a power of 2
    long logn = log2(n);
    // Rearrange the elements of x according to the bit-reversed order
    long i;
    // Create an array of threads
    pthread_t threads[num_threads];
    // Create an array of inputs to fft_helper_multi
    fft_helper_multi_input inputs[num_threads];
    // Perform the butterfly operations in parallel
    long ThreadLoad;
    if(logn % num_threads == 0)
    {
        ThreadLoad = logn/num_threads;
    }
    else
    {
        ThreadLoad = logn/num_threads + 1;
    }
    //= log2(n)/num_threads;
    long count = 0;
    long test = n/num_threads;
        // Divide the iterations into chunks and execute them in parallel
    if (pthread_barrier_init(&barrier1, NULL, num_threads)) {
    printf("Could not create a barrier\n");
    } 
    for (i = 0; i < num_threads; i++) {
        // Initialize the inputs to fft_helper_multi
        inputs[i].n = n;
        //inputs[i].x = x + (start * sizeof(cplx));
        inputs[i].x = x;
        inputs[i].end = (i+1)*(n/num_threads);
        inputs[i].start = i*(n/num_threads);
        inputs[i].totalrun = n/num_threads;
        inputs[i].jprotect = test/num_threads;
        inputs[i].thread_id = i;
        inputs[i].num_threads = num_threads;
        threadcounter += 1;
        // Create a new thread to execute the chunk
        pthread_create(&threads[i], NULL, fft_helper_multi, &inputs[i]);
        count++;
    }
        // Wait for all threads to finish before continuing to the next iteration
        for (i = 0; i < num_threads; i++) {
            pthread_join(threads[i], NULL);
        }
    }

// Test the fft function with a simple example
long main() {
  double multi_time = 0;
  printf("Number of iters: %d, size N: %d, Number of Threads: %d\n", num_iters, N, num_threads);

// Perform the GOOD FFT multi threaded on the input vector
for(long i = 0; i < num_iters; i++)
{
  for(long i = 0; i < N; i++)
  {
    in_x[i] = i;
  }
  clock_gettime(CLOCK_REALTIME, &start);
  fft_parallel(in_x, N, num_threads);
  clock_gettime(CLOCK_REALTIME, &end);
  multi_time += interval(start, end);
}
printf("Multi Time: %f\n", multi_time/num_iters);
multi_time = 0;


// Perform the serial FFT multi threaded on the input vector
  for(long i = 0; i < num_iters; i++)
  {
    for(long i = 0; i < N; i++)
    {
      in_x[i] = i;
    }
    clock_gettime(CLOCK_REALTIME, &start);
    fft(in_x, N);
    clock_gettime(CLOCK_REALTIME, &end);
    multi_time += interval(start, end);
  }
  printf("Serial Time: %f\n", multi_time/num_iters);
  multi_time = 0;

// Perform the serial V2 FFT multi threaded on the input vector
  // for(long i = 0; i < num_iters; i++)
  // {
  //   double x[N];  // Real part of input signal
  //   double y[N];  // Imaginary part of input signal
  //   long j;
  //   for(j = 0; j < N; j++)
  //   {
  //     x[j] = j;
  //     y[j] = 0;
  //   }
  //   clock_gettime(CLOCK_REALTIME, &start);
  //   fft_serialV2(N, x, y);
  //   clock_gettime(CLOCK_REALTIME, &end);
  //   multi_time += interval(start, end);
  // }
  // printf("Serial V2 Time: %f\n", multi_time/num_iters);
  // multi_time = 0;








//comment out below this for high runs____________________________________

// Perform the FFT multi threaded on the input vector via k loop(bad)
  //for(long i = 0; i < num_iters; i++)
  //{
  //  for(long i = 0; i < N; i++)
  //  {
  //    in_x[i] = i;
  //  }
  //  clock_gettime(CLOCK_REALTIME, &start);
  //  fft_bad(in_x, N, num_threads);
  //  clock_gettime(CLOCK_REALTIME, &end);
  //  multi_time += interval(start, end);
  //}
  //printf("multi(j-threaded) Time: %f\n", multi_time/num_iters);
  //multi_time = 0;

  // Perform the FFT multi threaded on the input vector via k loop(bad)
  //for(long i = 0; i < num_iters; i++)
  //{
  //  for(long i = 0; i < N; i++)
  //  {
  //    in_x[i] = i;
  //  }
  //  clock_gettime(CLOCK_REALTIME, &start);
  //  fft_k_thready(in_x, N, num_threads);
  //  clock_gettime(CLOCK_REALTIME, &end);
  //  multi_time += interval(start, end);
  //}
  //printf("multi(k-threaded bad) Time: %f\n", multi_time/num_iters);
  //multi_time = 0;

  return 0;
}