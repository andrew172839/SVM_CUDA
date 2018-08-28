SVM_CUDA is a GPU acceleration version for LIBSVM's cross validation

To run this program, CUDA 8.0 or other versions need to be preinstalled

Compile with CUDA 8.0:

g++ -c -o svm_revised.o svm_revised.cpp

cc -c -o train.o train.c

"/usr/local/cuda-8.0"/bin/nvcc -ccbin g++ -o train_gpu svm_revised.o train.o -L/usr/local/cuda-8.0/lib64 -lcublas -lcudart
