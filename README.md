## proxCSL

This repository contains the implementation of our method for distributed sparse logistic regression with proximal CSL updates. This method is published in our paper "High-Dimensional Distributed Sparse Classification with Scalable Communication-Efficient Global Updates" in KDD 2024.

In addition, we include our implementations of all baselines used in the paper: LIBLINEAR, Naive Averaging, OWA, sCSL, and sDANE. All methods are implemented in C++ and use the Armadillo and liblbfgs libraries.

For Armadillo installation, refer to: https://arma.sourceforge.net/

For liblbfgs refer to: 
https://www.chokkan.org/software/liblbfgs/

Additional dependencies will be needed. For example, if using Ubuntu or Debian, install dependencies with:

```
sudo apt-get install liblbfgs-dev libopenblas-dev g++ make
```

or, if on OS X, make sure that a C++ compiler is installed and run

```
brew install liblbfgs openblas
```

Once dependencies are installed, the following two targets are available:

 * `make single`: build programs to test proxCSL and competitors in the multicore single-node setting
 * `make mpi`: build MPI-enabled programs for distributed usage

Each program will print the required options.
Instead of running once on a given dataset, each program (e.g. `sweep_acowa`) will sweep a variety of lambda values on the same dataset.
In this way, performance for different numbers of nonzeros can be given.

For example, try running OWA, sCSL, and proxCSL on the included newsgroups dataset:

```
./sweep_owa data/newsgroups.svm owa.csv 1 -6 1 21 256
./sweep_csl data/newsgroups.svm csl.csv 1 -6 1 21 256 2 0
./sweep_prox_csl data/newsgroups.svm pcsl.csv 1 -6 1 21 256 2 0 1
```

All datasets should be in libsvm format.
