## Prompt
Write a short parallel program, in your language choice, using either MPI, OpenMP, OpenACC, OpenMP offload, or CUDA. Then push your code into a public GitHub repository. Please provide a link to your repository and write an explanation of the code.



## Explanation

This short parallel program implements the beginnings of a particle-in-cell (PIC) code. In particular, we implement a simplified version of what is typically the most computationally expensive routine in a PIC code. 

For our simplified PIC code, two data structures are implemented. We have (1) a 1D mesh grid with `nx` cells, and (2) an array of particles with continuous positions, which live on top of the mesh grid. We implement a CUDA kernel to loop through the particles and deposit their "charge" onto the grid — essentially, we create a histogram. In a PIC code, particles are not "point particles," rather they have a finite width, and some spatial shape. Here we use "linear" particle shapes — this means that particles can be thought of as being uniformly distributed over the width of one cell. Therefore depending on the particle's position in the cell, it deposits some fraction of it's charge into their current cell, and the rest into a neighboring cell (either the upper or lower neighbor on the grid).

Parallelism in the CUDA kernel is achieved by assigning one thread to one particle, and processing BLOCK_SIZE (the number of threads in a CUDA threadblock) number of particles at a time until all particles have been processed. I.e.

```
  part_idx = thread_index + block_index*threads_per_block
  while (part_idx < n_particles){
		deposit charge
    part_idx += BLOCK_SIZE;
  }
```

While this method of parallelizing the work is natural, and should good performance, this kernel is not "embarrassingly parallel." Since particles can have any position on the grid, it is possible that two threads will try to deposit charge onto the same grid cell at the same time. Therefore, to avoid memory collisions, an atomic operation is necessary when depositing charge onto the grid.



## Building the code 

The build system is GNU make. The makefile is (slightly) modified from [NVIDIA's cuda samples](https://github.com/NVIDIA/cuda-samples), which was used because it contains some nice logic to build on a variety of systems. On most systems, assuming that one has installed the NVIDIA CUDA toolkit, the code can be built by simply typing ``Make``.
