<!-- # Todo list of Project -->

### Issue and proposed solution
* CUDA
  * issue:
    1. __SVD performance:__ cupy > Pytorch GPU $\approx$ Pytorch CPU > numpy, GPU case on single card, CPU case on all 16 cores. 
    2. __cupy__ could not deal with large dataset, e.g. could not solve 2000000*1400 (double) of current task
    3. __Pytorch__ Pytorch uses Magma as svd backend on GPU (see https://discuss.pytorch.org/t/torch-svd-is-slow-in-gpu-compared-to-cpu/10770/2). It could deal more data than cupy on GPU, at least 2000000*1400 (double). Magma is able do computation gpu and also part of computation on cpu. That might be the reason of Pytorch can handle more data, as well as the reason SVD speed worse than cupy due to its non pure GPU calculation.
    4. __Tensorflow?__ current unable to detect all devices(only one cpu core and one GPU core detected), hence test is not performed. 
  * possible solution
    * __Batch svd__: [SVD using Pytorch](https://pytorch.org/docs/stable/torch.html#torch.svd) just calls function ```torch.svd(x), ``` , if ```x``` is 2d, e.g. ```(x, y)```, it do normal SVD. if ```x``` is 3d, e.g. ```(n, x, y)```. It would do batch SVD as n independent normal SVD, not like batch Gradient Descent, to split one normal SVD to blocks and feed to GPU in sequence to overcome the too large dataset issue .
    * __Random SVD and compressive Sensing__  Both are techniques to reduce dataset to feed to SVD. To make single card feasible.  
* Criteria of performance
  * L1, and L2 norm to compare by DMD recon-structured data. 
     
#### Environment
Kanon at [AER@TUM](https://www.mw.tum.de/aer/startseite/) 
* __cupy__ [cupy](https://docs-cupy.chainer.org/en/stable/) is kind of 'GPU version' of numpy which has very similar    syntax of numpy. It calls cuSolver of CUDA to do Linear Algebra.      
* __Magema__ Pytorch uses [Magma](https://icl.utk.edu/magma/software/index.html) for svd on GPU.  
* __python__ version: 3.6.6 by pip, 3.7.2 by miniconda 
* __julia__ optional if necessary
* __CUDA__ 10.0 sudo system default, but has issue in SVD for cupy. Hence 10.1 installed and without root used. 
* __Magma__ 2.5 and with support CUDA 101 installed by conda-forge
* __Pytorch__ 1.6 build from source with Magma 
* __Tensorflow__ 2.1 installed by conda-forge
* __dask__ latest version installed by pip


## Project structure 
### Part 1 DMD
- [x] Standard DMD
- [x] Multi-resolution DMD
- [x] compressed sensing DMD: compressing data for limit memory of gpu 
- [ ] Random DMD: save computation resource
- [ ] Extended DMD
### Part 2 Koopman




