#include<bits/stdc++.h>
#include<dirent.h>

#define sz(x) static_cast<int>((x).size())
using namespace std;

__global__ void convolve_parallel(float *input_cuda, float *kernel_cuda, float *output_cuda, int n, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < n - k + 1 && idy < n - k + 1) {
        float sum = 0.0;
        for (int x = 0; x < k; x++) {
            for (int y = 0; y < k; y++) {
                sum += input_cuda[(idx + x) * n + (idy + y)] * kernel_cuda[x * k + y];
            }
        }
        output_cuda[idx * (n - k + 1) + idy] = sum;
    }
}

__global__ void activate_parallel(float *input_cuda, float *output_cuda, int n, int m, int type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < n && idy < m) {
        int index = idx * n + idy;
        if (type == 0) {
            output_cuda[index] = max(0.0, input_cuda[index]);
        } else if (type == 1) {
            output_cuda[index] = tanhf(input_cuda[index]);
        }
    }
}

__global__ void maxpool_parallel(float *input_cuda, float *output_cuda, int n, int pool_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int output_size = n - pool_size + 1;
    if (row < output_size && col < output_size) {
        float maxVal = input_cuda[row*n+col];
        for (int i = 0; i < pool_size; ++i) {
            for (int j = 0; j < pool_size; ++j) {
                maxVal = fmaxf(maxVal, input_cuda[(row + i) * n + (col + j)]);
            }
        }
        output_cuda[row * output_size + col] = maxVal;
    }
}


__global__ void avgpool_parallel(float *input, float *output, int n, int pool_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int output_size = n - pool_size + 1;
    if (row < output_size && col < output_size) {
        float sum = 0.0;
        for (int i = 0; i < pool_size; ++i) {
            for (int j = 0; j < pool_size; ++j) {
                sum += input[(row+i)*n + (col+j)];
            }
        }
        output[row * output_size + col] = sum / (pool_size * pool_size);
    }
}

__global__ void normalize_sigmoid(float *input_cuda, float *output_cuda, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output_cuda[idx] = 1/(1+exp(-input_cuda[idx]));
}

void normalize_softmax(float *input, float *output, int n){
    float total = 0.0;
    for(int i=0;i<n;++i) total+=(float)exp(input[i]);
    for(int i=0;i<n;++i) output[i] = exp(input[i])/total;
}

int main(int argc, char* argv[])
{
    stringstream ss;
    for(int i=1;i<argc;++i){
        ss << argv[i] << " ";
    }
    int work;
    ss >> work;
    if (work == 1){
        int n, m , p;
        ss >> n >> m >> p;
        // n is input size
        // m is kernel size
        // p is padding
        int output_size = n + 2*p - m + 1;

        float* input = new float[(n + 2*p) * (n + 2*p)];
        float* kernel = new float[m*m];
        float* output = new float[output_size*output_size];

        for (int i = p; i < n + p; ++i) {
            for (int j = p; j < n + p; ++j) {
                ss >> input[i*(n+2*p) + j];
            }
        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                ss >> kernel[i*m+j];
            }
        }

        float* input_cuda;
        float* kernel_cuda;
        float* output_cuda;

        cudaMalloc(&input_cuda, (n + 2*p) * (n + 2*p) * sizeof(float));
        cudaMalloc(&kernel_cuda, m*m*sizeof(float));
        cudaMalloc(&output_cuda, output_size*output_size*sizeof(float));

        cudaMemcpy(input_cuda, input, (n + 2*p) * (n + 2*p) * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(kernel_cuda, kernel, m * m * sizeof(float), cudaMemcpyHostToDevice);

        dim3 num_blocks((output_size + 15) / 16, (output_size + 15) / 16);
        dim3 threads_per_block(16,16);

        convolve_parallel<<<num_blocks, threads_per_block>>> (input_cuda, kernel_cuda, output_cuda, n +2*p, m);

        cudaMemcpy(output, output_cuda, output_size*output_size*sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                cout << output[i*output_size + j] << " ";
            }
            cout << endl;
        }

        cudaFree(&input_cuda);
        cudaFree(&kernel_cuda);
        cudaFree(&output_cuda);
        delete input;
        delete output;
        delete kernel;
    }
    else if (work == 2){
        int activation_type;
        ss >> activation_type;
        int n, m;
        ss >> n >> m;
        float* input = new float[n * m];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                ss >> input[i*m+j];
            }
        }
        float* output = new float[n * m];
        float *input_cuda, *output_cuda;
        int output_size = n*m;
        cudaMalloc(&input_cuda, n * m * sizeof(float));
        cudaMalloc(&output_cuda, n * m * sizeof(float));
        cudaMemcpy(input_cuda, input, n * m * sizeof(float), cudaMemcpyHostToDevice);

        dim3 num_blocks((output_size + 15) / 16, (output_size + 15) / 16);
        dim3 threads_per_block(16,16);

        if (activation_type == 0){
            activate_parallel<<<num_blocks, threads_per_block>>>(input_cuda, output_cuda, n, m, 0);
        }
        else if (activation_type == 1){
            activate_parallel<<<num_blocks, threads_per_block>>>(input_cuda, output_cuda, n, m, 1);
        }

        cudaMemcpy(output, output_cuda, n * m * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < n; i++){
            for (int j = 0; j < m; j++){
                cout << output[i * m + j] << " ";
            }
            cout << endl;
        }
        cudaFree(&input_cuda);
        cudaFree(&output_cuda);
        delete input;
        delete output;
    }
    else if (work == 3){
        int pool_type;
        ss >> pool_type;

        int pool_size;
        ss >> pool_size;

        int n;
        ss >> n;
        
        float* input = new float[n*n];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                ss >> input[i * n + j];
            }
        }

        int output_size = n - pool_size + 1;
        float* output = new float[output_size * output_size];

        float *input_cuda, *output_cuda;
        cudaMalloc(&input_cuda, n * n * sizeof(float));
        cudaMalloc(&output_cuda, output_size * output_size * sizeof(float));
        cudaMemcpy(input_cuda, input, n * n * sizeof(float), cudaMemcpyHostToDevice);

        dim3 num_blocks((output_size + 15) / 16, (output_size + 15) / 16);
        dim3 threads_per_block(16,16);

        if (pool_type == 0){
            maxpool_parallel<<<num_blocks, threads_per_block>>>(input_cuda, output_cuda, n, pool_size);
        }
        else if (pool_type == 1){
            avgpool_parallel<<<num_blocks, threads_per_block>>>(input_cuda, output_cuda, n,pool_size);
        }

        cudaMemcpy(output, output_cuda, output_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                cout << output[i * output_size + j] << " ";
            }
            cout << endl;
        }
        cudaFree(&input_cuda);
        cudaFree(&output_cuda);
        delete input;
        delete output;
    }
    else if (work == 4){
        int normalization_type;
        ss >> normalization_type;
        int n = argc - 3;
        float* input = new float[n];
        for (int i = 0; i < n; ++i) {
            ss >> input[i];
        }
        float* output = new float[n];
        int output_size = n;
        
        float *input_cuda, *output_cuda;
        cudaMalloc(&input_cuda, n * sizeof(float));
        cudaMalloc(&output_cuda, n * sizeof(float));
        cudaMemcpy(input_cuda, input, n * sizeof(float), cudaMemcpyHostToDevice);

        dim3 num_blocks((output_size + 15) / 16, (output_size + 15) / 16);
        dim3 threads_per_block(16,16);
        if (normalization_type == 0){
            normalize_sigmoid<<<num_blocks, threads_per_block>>>(input_cuda, output_cuda, n);
            cudaMemcpy(output, output_cuda, output_size * sizeof(float), cudaMemcpyDeviceToHost);
        }
        else if (normalization_type == 1){
            normalize_softmax(input, output, n);
        }

        for (int i = 0; i < output_size; ++i) {
            cout<<output[i]<<" ";
        }
        cout<<endl;
        cudaFree(&input_cuda);
        cudaFree(&output_cuda);
        delete input;
        delete output;
    }
}
