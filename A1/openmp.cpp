#include <iostream>
#include <cmath>
#include <random>
#include <omp.h>
#include <chrono>

using namespace std;

int thread_count;
int n;
double **A, **L, **U;
int *pi;

void luDecomposition() {
    // Perform LU decomposition  
    for (int k = 0; k < n; ++k) {
        double maxVal = 0.0;
        int maxIndex = k;
        try
        {
            for (int i = k; i < n; ++i) {
                if (abs(A[i][k]) > maxVal) {
                    maxVal = abs(A[i][k]);
                    maxIndex = i;
                }
            }  
            if (maxVal == 0.0) {
                throw maxVal;
            }
        }
        catch(...)
        {
            printf("The input matrix is singular\n");
            terminate();
        }

        swap(pi[k], pi[maxIndex]);
        for(int i=0;i<n;++i){
            swap(A[k][i],A[maxIndex][i]);
        }
        for(int i=0;i<k;++i){
            swap(L[k][i],L[maxIndex][i]);
        }

        U[k][k] = A[k][k];
        
#       pragma omp parallel for num_threads(thread_count)
        for(int i=k+1;i<n;++i){
            L[i][k] = A[i][k] / U[k][k];
            U[k][i] = A[k][i]; 
        }
#       pragma omp parallel for num_threads(thread_count)
        for (int i = k+1; i < n; ++i) {
            for (int j = k+1; j < n; ++j) {
                A[i][j] = A[i][j] - (L[i][k] * U[k][j]);
            }
        }
    }
}

// Function to initialize a matrix with uniform random numbers
void initializeMatrix() {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(0.0, 1.0);

    for(int i=0;i<n;++i){
        for(int j=0;j<n;++j){
            A[i][j] = dis(gen);
        }
    }
    // A = {{1,2,3},{1,2,3},{1,2,3}};
    for (int i = 0; i < n; ++i) {
        L[i][i] = 1.0;
        pi[i] = i;
    }
}

// double computeL21Norm(double** PA, double** LU) {
//     double norm = 0.0;

//     for (int j = 0; j < n; ++j) {
//         double colNorm = 0.0;
//         for (int i = 0; i < n; ++i) {
//             colNorm += pow(PA[i][j] - LU[i][j], 2);
//         }
//         norm += sqrt(colNorm);
//     }

//     return norm;
// }

int main(int argc, char *argv[]) {
    
    if(argc != 3)
    {
        printf("Invalid input format\n");
        return 0;
    }
    
    thread_count = strtol(argv[2],NULL,10); //number of threads
    n = strtol(argv[1],NULL,10); //size of the matrix A

    A = new double*[n];
    L = new double*[n];
    U = new double*[n];
    pi = new int[n];
    for (int i = 0; i < n; ++i) {
        A[i] = new double[n];
        L[i] = new double[n];
        U[i] = new double[n];
    }
    
    initializeMatrix();

    // double** A_old = new double*[n];
    // for (int i = 0; i < n; ++i) {
    //     A_old[i] = new double[n];
    //     for (int j = 0; j < n; ++j) {
    //         A_old[i][j] = A[i][j];
    //     }
    // }

    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    luDecomposition();
        
    auto end = std::chrono::high_resolution_clock::now(); 
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - now).count();

    std::cout << "Time taken: " << ((double) duration)/1000000 << " seconds" << std::endl;


    // //printing A
    std::cout<<"\nPrinting value of A:\n";
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
        {
            std::cout<<A[i][j]<<" ";
        }
        std::cout<<endl;
    }

    //printing L
    std::cout<<"\nPrinting value of L:\n";
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
        {
            std::cout<<L[i][j]<<" ";
        }
        std::cout<<endl;
    }

    //printing R
    std::cout<<"\nPrinting value of U:\n";
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
        {
            std::cout<<U[i][j]<<" ";
        }
        std::cout<<endl;
    }

    //printing pi
    std::cout<<"\nPrinting value of pi:\n";
    for(int i=0;i<n;i++)
    {
        std::cout<<pi[i]<<" ";
    }

    // double** LU = new double*[n];
    // for (int i = 0; i < n; ++i) {
    //     LU[i] = new double[n];
    //     for (int j = 0; j < n; ++j) {
    //         double sum = 0.0;
    //         for (int k = 0; k <= min(i, j); ++k) {
    //             sum += L[i][k] * U[k][j];
    //         }
    //         LU[i][j] = sum;
    //     }
    // }

    // double** PA = new double*[n];
    // for (int i = 0; i < n; ++i) {
    //     PA[i] = new double[n];
    //     for (int j = 0; j < n; ++j) {
    //         PA[i][j] = A_old[pi[i]][j];
    //     }
    // }

    // double norm = computeL21Norm(PA, LU);
    // cout << "L2,1 norm of the residual: " << norm << endl;
    for (int i = 0; i < n; ++i) {
        delete[] A[i];
        delete[] L[i];
        delete[] U[i];
        // delete[] A_old[i];
        // delete[] LU[i];
        // delete[] PA[i];
    }
    delete[] A;
    delete[] L;
    delete[] U;
    // delete[] A_old;
    // delete[] LU;
    // delete[] PA;

    return 0;
}
