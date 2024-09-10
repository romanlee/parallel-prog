#include <stdio.h>
#include <assert.h>
#include <vector>
#include <iostream>
#include <random>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// // uncomment to enable debug mode
// #define __DEBUG__

// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
void checkError(cudaError_t code, char const * func, const char *file, const int line,
                bool abort) {
  if (code != cudaSuccess) {
    const char * errorMessage = cudaGetErrorString(code);
    fprintf(stderr, "CUDA error returned from \"%s\" at %s:%d, Error code: %d (%s)\n",
            func, file, line, code, errorMessage);
    if (abort) {
      cudaDeviceReset();
      exit(code);
    }
  }
}
#define chkErr(val) checkError((val), #val, __FILE__, __LINE__, true)


// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
void checkLastError(char const * func, const char *file, const int line, bool abort) {
  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    const char * errorMessage = cudaGetErrorString(code);
    fprintf(stderr, "CUDA error returned from \"%s\" at %s:%d, Error code: %d (%s)\n",
            func, file, line, code, errorMessage);
    if (abort) {
      cudaDeviceReset();
      exit(code);
    }
  }
}
#define chkLastErr(func) checkLastError(func, __FILE__, __LINE__, true)


// --------------------------------------------------------------------------------------
// Linear particle shape function
// --------------------------------------------------------------------------------------
__forceinline__ __device__  void spline( float x, float *s){

  s[0] = 0.5 + x;
  s[1] = 0.5 - x;

}


// --------------------------------------------------------------------------------------
// Accumulate particle charge onto grid
// --------------------------------------------------------------------------------------
template <int BLOCK_SIZE>
__global__ void deposit(int nx, float *grid, int n_par, int *ix, float *x) {

  int bx = blockIdx.x;
  int tx = threadIdx.x;

  // array to store spline for each particle
  float s[2];

  // NOTE: indexing particles in this way (with tx varying quickest) achieves coalesced 
  // memory access
  int part_idx = bx*BLOCK_SIZE + tx;

  while (part_idx < n_par){

    float x_part = x[part_idx];

    // get spline weights for current particle
    spline( x_part, s );

    // get lower cell index
    // remember x can be between -.5 and .5, so floor(x) gives either -1 or 0
    int lower_cell_idx = floor(x_part) + ix[part_idx];

    // deposit charge onto the grid
    if (lower_cell_idx >= 0){
      atomicAdd( &grid[ lower_cell_idx ], s[0] );
    }
    if (lower_cell_idx+1 <= nx){
      atomicAdd( &grid[ lower_cell_idx+1 ], s[1] );
    }

    // increment particle index
    part_idx += BLOCK_SIZE;

  }

}


// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
int main(int argc, char **argv) {


  // number of cells in our 1D grid
  int nx = 16;

  // particles per cell
  int ppc = 32;
  int n_par = nx * ppc;

  // For the illustrative purposes of this example we just use one block
  // A realistic PIC code would have more particles and would require more blocks to 
  // achieve good performance
  int nblocks = 1;

  // define particle arrays
  int *h_ix;
  float *h_x;

  // create grid onto which particle charge will be accumulated
  float *h_charge;


  // allocate host mem
  int sizeof_ix = n_par * sizeof(int);
  h_ix = (int*) malloc(sizeof_ix);

  int sizeof_x = n_par * sizeof(float);
  h_x = (float*)malloc(sizeof_x);

  int sizeof_charge = nx * sizeof(float);
  h_charge = (float*)malloc(sizeof_charge);

  
  // initialize particle array and charge grid
  //   particles are initialized with `ppc` particles per cell and position within the
  //   cell randomized
  int part_idx = 0;
  for (int cell_idx=0; cell_idx<nx; cell_idx++){

    h_charge[cell_idx] = 0;

    for (int j=0; j<ppc; j++){
      // randomly generate float-component of particle position, between -0.5 and 0.5
      float x = static_cast <float> (rand()) /( static_cast <float> (RAND_MAX)) - 0.5;

      h_ix[part_idx] = cell_idx;
      h_x[part_idx]  = x;

      // std::cout << "part_idx: " << part_idx  << "  ix: "<< cell_idx << "  x: " << x << '\n';

      part_idx++;
    }

  }

#ifdef __DEBUG__
  // Validate initialized particle arrays
  // Given more time, could write some more sophisticated/automatic testing functions
  for ( int i=0; i<n_par; i++ ){
    std::cout << "part_idx: " << i << "  ix: "<< h_ix[i] << "  x: " << h_x[i] << '\n';
  }
  std::cout << "\n";
#endif


  // allocate device memory
  int *d_ix;
  float *d_x;
  float *d_charge;
  chkErr(cudaMalloc( reinterpret_cast<void **>(&d_ix), sizeof_ix ));
  chkErr(cudaMalloc( reinterpret_cast<void **>(&d_x), sizeof_x ));
  chkErr(cudaMalloc( reinterpret_cast<void **>(&d_charge), sizeof_charge ));


  // copy host memory to device
  chkErr(cudaMemcpy(d_ix, h_ix, sizeof_ix, cudaMemcpyHostToDevice));
  chkErr(cudaMemcpy(d_x, h_x, sizeof_x, cudaMemcpyHostToDevice));
  chkErr(cudaMemcpy(d_charge, h_charge, sizeof_charge, cudaMemcpyHostToDevice));


  // simulation loop --- well, in a real code this would be a loop, but here we just 
  // implement one iteration (and one routine within that iteration)

  // 1. update particle velocity and advance particle position

  // 2. deposit charge onto grid
  //    In a real pic code, one would deposit *current* onto the grid in a similar fashion
  //    NOTE: we hard-code block size 32. One could play around with this to achieve best performance
  deposit <32> <<< nblocks, 32 >>> (nx, d_charge, n_par, d_ix, d_x);

  chkLastErr("deposit");
#ifdef __DEBUG__
  // improves experience with printing from kernels
  chkErr(cudaDeviceSynchronize());
#endif

  // 3. In a real pic code, the current deposited in the previous step would be used to
  //    solve for E and B fields using finite differences. These fields would then be 
  //    used to advance particle position at the top of the loop.


  // check the output
  // One expects that the average charge should be just below ppc
  chkErr(cudaMemcpy(h_charge, d_charge, sizeof_charge, cudaMemcpyDeviceToHost));

  float ave_charge = 0;
  for ( int i=0; i<nx; i++ ){
    ave_charge += h_charge[i];
    std::cout << "cell idx: " << i << "    charge: " << h_charge[i] << "\n";
  }
  ave_charge /= nx;
  std::cout<< "average charge: " << ave_charge << "\n";


  // cleanup
  chkErr(cudaFree(d_ix));
  chkErr(cudaFree(d_x));
  chkErr(cudaFree(d_charge));

  free(h_ix);
  free(h_x);
  free(h_charge);
  

}
