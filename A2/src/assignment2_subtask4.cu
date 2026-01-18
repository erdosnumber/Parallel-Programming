#include <bits/stdc++.h>
using namespace std;

// ofstream our_output;

// 2d input * 3d kernel => 3d output
__global__ void convolve_cuda(float *input_cuda, float *kernel_cuda, float *output_cuda, int input_size, int kernel_size, int depth, float *bias)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    if ((idx < input_size - kernel_size + 1) && (idy < input_size - kernel_size + 1) && (idz < depth))
    {
        float sum = 0.0;
        for (int x = 0; x < kernel_size; x++)
        {
            for (int y = 0; y < kernel_size; y++)
            {
                sum += input_cuda[(idx + x) * input_size + (idy + y)] * kernel_cuda[idz * kernel_size * kernel_size + x * kernel_size + y];
            }
        }
        output_cuda[idz * (input_size - kernel_size + 1) * (input_size - kernel_size + 1) + idx * (input_size - kernel_size + 1) + idy] = (sum + bias[idz]);
    }
}

// 3d input, 4d kernel => 3d output
__global__ void convolve_cuda_3d_channel(float *input_cuda, float *kernel_cuda, float *output_cuda, int input_size, int kernel_size, int depth, int channels)
{

    // depth is 20 and channels is 50

    // blockDim.z is going to be 20*50
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    // idz is from 0 to 999
    // depth is the input depth and channels is the number of kernels

    int self_depth = idz % depth;
    int channel_id = idz / depth;
    int output_size = input_size - kernel_size + 1;
    // channels is 50 (for now), idz_channel lies between (0,49)
    // idz_index lies in (0,19)

    if ((idx < input_size - kernel_size + 1) && (idy < input_size - kernel_size + 1) && (channel_id < channels))
    {
        float sum = 0.0;
        for (int x = 0; x < kernel_size; x++)
        {
            for (int y = 0; y < kernel_size; y++)
            {
                sum += input_cuda[(input_size * input_size * self_depth) + (input_size * (idx + x)) + (idy + y)] * kernel_cuda[(kernel_size * kernel_size * depth * channel_id) + (kernel_size * kernel_size * self_depth) + (kernel_size * x) + y];
            }
        }
        // atomic add
        atomicAdd(&output_cuda[(output_size * output_size * channel_id) + (output_size * idx) + idy], sum);
    }
}

// softmax
__global__ void output_probability(float *input_cuda, float *output_cuda, int n)
{
    int idx = threadIdx.x;
    if(idx<1){
        float total = 0.0;
        for (int i = 0; i < n; ++i)
            total += (float)exp(input_cuda[i]);
        for (int i = 0; i < n; ++i)
            output_cuda[i] = 100*exp(input_cuda[i]) / total;
    }
}

// relu activation
__global__ void activate_parallel(float *input_cuda, float *output_cuda, int n, int type)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        int index = idx;
        // 0 is for relu, 1 is for tanh
        if (type == 0)
        {
            output_cuda[index] = fmaxf(0.0, input_cuda[index]);
        }
        else if (type == 1)
        {
            output_cuda[index] = tanhf(input_cuda[index]);
        }
    }
}

// maxpooling assuming stride is 2 as per the details.txt
__global__ void maxPool_cuda(float *input_cuda, float *output_cuda, int input_size, int pool_size, int num_filters)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int filter_id = blockIdx.z * blockDim.z + threadIdx.z;

    int output_size = input_size / pool_size;

    if ((row < output_size) && (col < output_size) && (filter_id < num_filters))
    {
        float maxVal = input_cuda[(filter_id * input_size * input_size) + (row * pool_size * input_size) + (col * pool_size)];
        for (int i = 0; i < pool_size; ++i)
        {
            for (int j = 0; j < pool_size; ++j)
            {
                maxVal = fmaxf(maxVal, input_cuda[(filter_id * input_size * input_size) + (input_size * (row * pool_size + i)) + (col * pool_size + j)]);
            }
        }
        output_cuda[(filter_id * output_size * output_size) + (output_size * row) + col] = maxVal;
    }
}

// reads the filename into input array
// not necessary convolution
void load_convolution_values(const char *filename, float *values, int num_values)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < num_values; ++i)
    {
        if (!(file >> values[i]))
        {
            std::cerr << "Error reading value at index " << i << " from file: " << filename << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    file.close();
}

// CONVULUTION LAYER 1
float *conv1_values;
float *conv1_filter;
float *conv1_bias;
float *conv1_filter_cuda;
float *conv1_bias_cuda;

void conv1_init()
{
    int k = 5;
    // int image_size = 28;
    // int output_size = 24; //(n-k+1)

    int num_filters = 20;
    int num_conv_values = 520;

    conv1_values = new float[num_conv_values];
    load_convolution_values("./weights/conv1.txt", conv1_values, num_conv_values);
    conv1_filter = new float[k * k * num_filters];
    conv1_bias = new float[num_filters];

    for (int i = 0; i < k * k * num_filters; i++)
        conv1_filter[i] = conv1_values[i];
    for (int i = 0; i < num_filters; ++i)
        conv1_bias[i] = conv1_values[k * k * num_filters + i];

    cudaMalloc(&conv1_filter_cuda, k * k * num_filters * sizeof(float));
    cudaMemcpy(conv1_filter_cuda, conv1_filter, k * k * num_filters * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&conv1_bias_cuda, num_filters * sizeof(float));
    cudaMemcpy(conv1_bias_cuda, conv1_bias, num_filters * sizeof(float), cudaMemcpyHostToDevice);
}

void conv1_finish()
{
    free(conv1_values);
    free(conv1_filter);
    free(conv1_bias);
    cudaFree(&conv1_filter_cuda);
    cudaFree(&conv1_bias_cuda);
}

float *conv2_values;
float *conv2_filter;
float *conv2_bias;
float *conv2_filter_cuda;
float *conv2_bias_cuda;

void conv2_init()
{
    int kernel_size = 5;
    // int input_size = 12;
    // int output_size = input_size - kernel_size + 1; // 8

    int num_filters_input = 20;
    int num_filters_output = 50;
    int num_conv_values = 25050;

    conv2_values = new float[num_conv_values];
    load_convolution_values("./weights/conv2.txt", conv2_values, num_conv_values);
    conv2_filter = new float[num_filters_input * num_filters_output * kernel_size * kernel_size];
    conv2_bias = new float[num_filters_output];

    for (int i = 0; i < (kernel_size * kernel_size * num_filters_input * num_filters_output); ++i)
        conv2_filter[i] = conv2_values[i];
    for (int i = 0; i < num_filters_output; ++i)
        conv2_bias[i] = conv2_values[kernel_size * kernel_size * num_filters_output * num_filters_input + i];

    cudaMalloc(&conv2_filter_cuda, num_filters_input * num_filters_output * kernel_size * kernel_size * sizeof(float));
    cudaMemcpy(conv2_filter_cuda, conv2_filter, num_filters_input * num_filters_output * kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&conv2_bias_cuda, num_filters_output * sizeof(float));
    cudaMemcpy(conv2_bias_cuda, conv2_bias, num_filters_output * sizeof(float), cudaMemcpyHostToDevice);
}

void conv2_finish()
{
    free(conv2_values);
    free(conv2_filter);
    free(conv2_bias);
    cudaFree(&conv2_filter_cuda);
    cudaFree(&conv2_bias_cuda);
}

float *fc1_values;
float *fc1_filter;
float *fc1_bias;
float *fc1_filter_cuda;
float *fc1_bias_cuda;

void fc1_init()
{
    int kernel_size = 4;
    // int input_size = 4;
    // int output_size = input_size - kernel_size + 1; // 1

    int num_filters_input = 50;
    int num_filters_output = 500;
    int num_conv_values = num_filters_output * kernel_size * kernel_size * num_filters_input + num_filters_output; // 400500

    fc1_values = new float[num_conv_values];
    load_convolution_values("./weights/fc1.txt", fc1_values, num_conv_values);
    fc1_filter = new float[num_filters_input * num_filters_output * kernel_size * kernel_size];
    fc1_bias = new float[num_filters_output];

    for (int i = 0; i < kernel_size * kernel_size * num_filters_input * num_filters_output; ++i)
        fc1_filter[i] = fc1_values[i];
    for (int i = 0; i < num_filters_output; ++i)
        fc1_bias[i] = fc1_values[kernel_size * kernel_size * num_filters_output * num_filters_input + i];

    cudaMalloc(&fc1_filter_cuda, kernel_size * kernel_size * num_filters_output * num_filters_input * sizeof(float));
    cudaMemcpy(fc1_filter_cuda, fc1_filter, kernel_size * kernel_size * num_filters_output * num_filters_input * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&fc1_bias_cuda, num_filters_output * sizeof(float));
    cudaMemcpy(fc1_bias_cuda, fc1_bias, num_filters_output * sizeof(float), cudaMemcpyHostToDevice);
}

void fc1_finish()
{
    free(fc1_values);
    free(fc1_filter);
    free(fc1_bias);
    cudaFree(&fc1_filter_cuda);
    cudaFree(&fc1_bias_cuda);
}
float *fc2_values;
float *fc2_filter;
float *fc2_bias;
float *fc2_filter_cuda;
float *fc2_bias_cuda;

void fc2_init()
{
    int kernel_size = 1;
    // int input_size = 1;
    // int output_size = input_size - kernel_size + 1; // 1

    int num_filters_input = 500;
    int num_filters_output = 10;
    int num_conv_values = num_filters_output * kernel_size * kernel_size * num_filters_input + num_filters_output; // 5010

    fc2_values = new float[num_conv_values];
    load_convolution_values("./weights/fc2.txt", fc2_values, num_conv_values);
    fc2_filter = new float[num_filters_input * num_filters_output * kernel_size * kernel_size];
    fc2_bias = new float[num_filters_output];

    for (int i = 0; i < kernel_size * kernel_size * num_filters_input * num_filters_output; ++i)
        fc2_filter[i] = fc2_values[i];
    for (int i = 0; i < num_filters_output; ++i)
        fc2_bias[i] = fc2_values[kernel_size * kernel_size * num_filters_output * num_filters_input + i];

    cudaMalloc(&fc2_filter_cuda, kernel_size * kernel_size * num_filters_output * num_filters_input * sizeof(float));
    cudaMemcpy(fc2_filter_cuda, fc2_filter, kernel_size * kernel_size * num_filters_output * num_filters_input * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&fc2_bias_cuda, num_filters_output * sizeof(float));
    cudaMemcpy(fc2_bias_cuda, fc2_bias, num_filters_output * sizeof(float), cudaMemcpyHostToDevice);
}

void fc2_finish()
{
    free(fc2_values);
    free(fc2_filter);
    free(fc2_bias);
    cudaFree(&fc2_filter_cuda);
    cudaFree(&fc2_bias_cuda);
}


__global__ void add_bias(float *bias_cuda, float *output_cuda, int output_size, int channels)
{
    int idx = threadIdx.x;
    int idy = threadIdx.y;
    int idz = blockIdx.x;
    // idz is from 0 to 999
    // depth is the input depth and channels is the number of kernels

    // idx and idy matrix dimensions and idz denotes depth
    if (idx < output_size && idy < output_size && idz < channels)
    {
        output_cuda[(output_size * output_size * idz) + (output_size * idx) + idy] = bias_cuda[idz];
    }
}

int main(int argc, char* argv[])
{
    if(atoi(argv[1])==0){
        cout<<"running subtask3 from subtask4"<<endl<<endl;
        system("make subtask3");
        system("./subtask3");
        return 0;
    }
    cout<<"reading kernels"<<endl;
    conv1_init();
    conv2_init();
    fc1_init();
    fc2_init();
    // our_output.open("our_output.txt");
 
    int number_images = 10000;

    long numstreams = 5;
    cudaStream_t streams[numstreams];
    for (int i = 0; i < numstreams; i++)
    {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    float *input_cuda_conv1;
    float *input_cuda_pool1;
    float *input_cuda_conv2;
    float *input_cuda_pool2;
    float *input_cuda_fc1;
    float *input_cuda_fc2;
    float *input_cuda_fc2_after;
    float *output_cuda;
    float *output_cuda_after;


    cout<<"allocating memory in gpu"<<endl<<endl;
    cudaMalloc(&input_cuda_conv1, 28 * 28 * sizeof(float) * number_images);
    cudaMalloc(&input_cuda_pool1, 20 * 24 * 24 * sizeof(float) * numstreams);
    cudaMalloc(&input_cuda_conv2, 20 * 12 * 12 * sizeof(float) * numstreams);
    cudaMalloc(&input_cuda_pool2, 50 * 8 * 8 * sizeof(float) * numstreams);
    cudaMalloc(&input_cuda_fc1, 50 * 4 * 4 * sizeof(float) * numstreams);
    cudaMalloc(&input_cuda_fc2, 500 * sizeof(float) * numstreams);
    cudaMalloc(&input_cuda_fc2_after, 500 * sizeof(float) * numstreams);
    cudaMalloc(&output_cuda, 10 * sizeof(float) * numstreams);
    cudaMalloc(&output_cuda_after, 10 * sizeof(float) * number_images);


    struct timespec start1, end1, start2, end2, start3, end3;

    // measuring processing time to read the images
    long timediff1;
    clock_gettime(CLOCK_MONOTONIC, &start1);
    cout<<"reading the images"<<endl;

    float *input = new float[28 * 28 * number_images];
    for (int i = 0; i < number_images; i++)
    {
        string filename = "./pre-proc-img/";
        stringstream ss;
        ss << i;
        string str = ss.str();
        for (int j = 0; j < 6 - str.length(); ++j)
        {
            filename += '0';
        }
        filename += str;
        filename += ".txt";
        load_convolution_values(filename.c_str(), input + i * 28 * 28, 28 * 28);
    }

    dim3 num_blocks_conv1(1, 1, 20);
    dim3 threads_per_block_conv1(24, 24, 1);

    dim3 num_blocks_pool1(1, 1, 20);
    dim3 threads_per_block_pool1(12, 12, 1);

    dim3 num_blocks_bias_conv2(50, 1, 1);
    dim3 threads_per_block_bias_conv2(8, 8, 1);

    dim3 num_blocks_conv2(1, 1, 1000);
    dim3 threads_per_block_conv2(8, 8, 1);

    dim3 num_blocks_bias_fc1(500, 1, 1);
    dim3 threads_per_block_bias_fc1(1, 1, 1);

    dim3 num_blocks_pool2(1, 1, 50);
    dim3 threads_per_block_pool2(4, 4, 1);

    dim3 num_blocks_relu_fc1(1, 1, 1);
    dim3 threads_per_block_relu_fc1(500, 1, 1);

    dim3 num_blocks_fc1(1, 1, 500 * 50);
    dim3 threads_per_block_fc1(1, 1, 1);

    dim3 num_blocks_bias_fc2(10,1,1);
    dim3 threads_per_block_bias_fc2(1,1,1);
    
    dim3 num_blocks1_fc2(1, 1, 10 * 500);
    dim3 threads_per_block1_fc2(1, 1, 1);


    cudaMemcpy(input_cuda_conv1, input, 28 * 28 * sizeof(float) * number_images, cudaMemcpyHostToDevice);


    clock_gettime(CLOCK_MONOTONIC, &end1);
    timediff1 = (end1.tv_sec - start1.tv_sec) * 1000 + (end1.tv_nsec - start1.tv_nsec) / 1000000;
    cout<<"reading completed"<<endl;
    std::cout << "Time taken for reading: " << timediff1 << " milliseconds" << std::endl << endl;

    // once images have been read, inference them
    cout<<"inference starting"<<endl;
    long timediff2;
    clock_gettime(CLOCK_MONOTONIC, &start2);

    for (int i = 0; i < number_images; i++)
    {
        convolve_cuda<<<num_blocks_conv1, threads_per_block_conv1, 0, streams[i % numstreams]>>>(input_cuda_conv1 + i * 28 * 28, conv1_filter_cuda, input_cuda_pool1 + (i % numstreams) * 20 * 24 * 24, 28, 5, 20, conv1_bias_cuda);
        
        maxPool_cuda<<<num_blocks_pool1, threads_per_block_pool1, 0, streams[i % numstreams]>>>(input_cuda_pool1 + (i % numstreams) * 20 * 24 * 24, input_cuda_conv2 + (i % numstreams) * 20 * 12 * 12, 24, 2, 20);
        
        add_bias<<<num_blocks_bias_conv2, threads_per_block_bias_conv2, 0, streams[i % numstreams]>>>(conv2_bias_cuda, input_cuda_pool2 + (i % numstreams) * 50 * 8 * 8, 8, 50);
        
        convolve_cuda_3d_channel<<<num_blocks_conv2, threads_per_block_conv2, 0, streams[i % numstreams]>>>(input_cuda_conv2 + (i % numstreams) * 20 * 12 * 12, conv2_filter_cuda, input_cuda_pool2 + (i % numstreams) * 50 * 8 * 8, 12, 5, 20, 50);
        
        maxPool_cuda<<<num_blocks_pool2, threads_per_block_pool2, 0, streams[i % numstreams]>>>(input_cuda_pool2 + (i % numstreams) * 50 * 8 * 8, input_cuda_fc1 + (i % numstreams) * 50 * 4 * 4, 8, 2, 50);
        
        add_bias<<<num_blocks_bias_fc1, threads_per_block_bias_fc1, 0, streams[i % numstreams]>>>(fc1_bias_cuda, input_cuda_fc2 + (i % numstreams) * 500, 1, 500);
        
        convolve_cuda_3d_channel<<<num_blocks_fc1, threads_per_block_fc1, 0, streams[i % numstreams]>>>(input_cuda_fc1 + (i % numstreams) * 50 * 4 * 4, fc1_filter_cuda, input_cuda_fc2 + (i % numstreams) * 500, 4, 4, 50, 500);
        
        activate_parallel<<<num_blocks_relu_fc1, threads_per_block_relu_fc1, 0, streams[i % numstreams]>>>(input_cuda_fc2 + (i % numstreams) * 500, input_cuda_fc2_after + (i % numstreams) * 500, 500, 0);

        add_bias<<<num_blocks_bias_fc2, threads_per_block_bias_fc2, 0, streams[i % numstreams]>>>(fc2_bias_cuda, output_cuda + (i % numstreams) * 10, 1, 10);
        
        convolve_cuda_3d_channel<<<num_blocks1_fc2, threads_per_block1_fc2, 0, streams[i % numstreams]>>>(input_cuda_fc2_after + (i % numstreams) * 500, fc2_filter_cuda, output_cuda + (i % numstreams) * 10, 1, 1, 500, 10);

        output_probability<<<1, 10, 0, streams[i % numstreams]>>>(output_cuda + (i % numstreams) * 10, output_cuda_after + (i * 10), 10);
    }

    cudaDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC, &end2);
    timediff2 = (end2.tv_sec - start2.tv_sec) * 1000 + (end2.tv_nsec - start2.tv_nsec) / 1000000;
    cout<<"inference complete"<<endl;
    std::cout << "Time taken for inference: " << timediff2 << " milliseconds" << std::endl << endl;

    for (int i = 0; i < numstreams; i++)
    {
        cudaStreamDestroy(streams[i]);
    }

    float *output = new float[10 * number_images];
    cudaMemcpy(output, output_cuda_after, 10 * sizeof(float) * number_images, cudaMemcpyDeviceToHost);

    cout<<"writing the outputs"<<endl;
    long timediff3;
    clock_gettime(CLOCK_MONOTONIC, &start3);

    for (int i = 0; i < number_images; i++)
    {
        string output_filename = "output/";
        stringstream ss;
        ss << i;
        string str = ss.str();
        for (int j = 0; j < 6 - str.length(); ++j)
        {
            output_filename += '0';
        }
        output_filename += str;
        output_filename += ".txt";

        // string output_filename = "test_output/";
        // stringstream ss;
        // ss << i;
        // string str = ss.str();
        // for (int j = 0; j < 6 - str.length(); ++j)
        // {
        //     output_filename += '0';
        // }
        // output_filename += str;

        ofstream file(output_filename.c_str());

        // float max_prob = 0.0;
        // int ans_number = 0;

        // for(int j=0;j<10;++j){
        //     if(output[i * 10 + j] > max_prob)
        //     {
        //         max_prob = output[i * 10 + j];
        //         ans_number = j;
        //     }
        // }

        // our_output<<output_filename<<" "<<ans_number<<endl;

        vector<pair<float, int> > probabilities;
        for (int j = 0; j < 10; ++j)
        {
            probabilities.push_back(make_pair(output[i * 10 + j], j));
        }
        sort(probabilities.rbegin(), probabilities.rend());
        for (int j = 0; j < 5; ++j)
        {
            file << std::fixed << std::setprecision(4);
            file << probabilities[j].first << " " << "class" << " " << probabilities[j].second << endl;
        }
        file.close();
    }

    clock_gettime(CLOCK_MONOTONIC, &end3);
    timediff3 = (end3.tv_sec - start3.tv_sec) * 1000 + (end3.tv_nsec - start3.tv_nsec) / 1000000;
    cout<<"writing completed"<<endl;
    std::cout << "Time taken for writing: " << timediff3 << " milliseconds" << std::endl << endl << endl;

    // this is the total time, reading + inference + writiing time
    cout<<"total time taken: "<<timediff1 + timediff2 + timediff3 << endl;

    // our_output.close();
    // Free all the variables
    cudaFree(&input_cuda_conv1);
    cudaFree(&input_cuda_pool1);
    cudaFree(&input_cuda_conv2);
    cudaFree(&input_cuda_pool2);
    cudaFree(&input_cuda_fc1);
    cudaFree(&input_cuda_fc2);
    cudaFree(&output_cuda);
    cudaFree(&output_cuda_after);

    free(input);
    free(output);
    conv1_finish();
    conv2_finish();
    fc1_finish();
    fc2_finish();
}
