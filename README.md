# cuda-uach

CUDA Scripts for "GPU Programming" classes in UACh: Matrix multiplication, reductions, fractals, and more.

## Content

- **JuliaSet:** Julia fractal implemented in C++ CUDA: display the fractal in terminal or it can be saved in a file to visualize in ParaView

- **ClassicMatmul:** matrix multiplication implemented for CPU(using OpenMP and analyzing transpose matrix effects) and CUDA(using SHared Memory too)

- **MatmulMultiGPU:**Matrix multiplication implemented for CPU(OpenMP) and CUDA(using multiGPU). It also includes mem.cu file which implements naive matmul considering Unified Memory.

- **ShuffleReduction**: Reduction implementation in CUDA using: __shfl__ functions, atomics, and omp reduce(for CPU).

- **Streams**: simple implementation of CUDA streams.

## References

- Sanders, J., & Kandrot, E. (2010). CUDA by example: an introduction to general-purpose GPU programming, portable documents. Addison-Wesley Professional.

- Wilt, N. (2013). The cuda handbook: A comprehensive guide to gpu programming. Pearson Education.

- Soyata, T. (2018). GPU Parallel Program Development Using CUDA. Chapman and Hall/CRC.






