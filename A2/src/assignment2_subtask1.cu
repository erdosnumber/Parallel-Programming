#include<bits/stdc++.h>
#define sz(x) static_cast<int>((x).size())
using namespace std;

vector<vector<float> > convolve(vector<vector<float> > &input,vector<vector<float> > &kernel,int padding)
{
    int n = sz(input),k = sz(kernel);

    //apply the padding
    vector<vector<float> > padded_input(n+2*padding,vector<float>(n+2*padding,0.0));
    for(int i=padding;i<n+padding;i++)
    {
        for(int j=padding;j<n+padding;j++)
        {
            padded_input[i][j] = input[i-padding][j-padding];
        }
    }

    input=padded_input;
    n = sz(input);

    //convolution
    vector<vector<float> > result(n-k+1,vector<float>(n-k+1,0.0));

    for(int i=0;i<n-k+1;i++)
    {
        for(int j=0;j<n-k+1;j++)
        {
            for(int x=0;x<k;x++)
            {
                for(int y=0;y<k;y++)
                {
                    result[i][j] += input[i+x][j+y]*kernel[x][y];
                }
            }
        }
    }

    return result;
}

vector<vector<float> > activate(vector<vector<float> > &input,string type)
{
    vector<vector<float> > output = input;
    int n=sz(input),m=sz(input[0]);

    if(type == "relu")
    {
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<m;j++) 
            {
                if(output[i][j]<=0) output[i][j]=0;
            }
        }
    }
    else if(type == "sigmoid")
    {
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<m;j++) output[i][j] = 1/(1+exp(-output[i][j]));
        }
    }
    else if(type == "tanh")
    {
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<m;j++) output[i][j] = tanh(output[i][j]);
        }
    }

    return output;
}

vector<vector<float> > max_pooling(vector<vector<float> >& input,int pool_size)
{
    int n=sz(input);
    int output_size = n - pool_size + 1;

    vector<vector<float> > result(output_size,vector<float>(output_size,0.0));

    for(int i=0;i<output_size;++i){
        for(int j=0;j<output_size;++j){
            result[i][j] = input[i][j];
            for(int k = i; k< i + pool_size ;++k){
                for(int h = j; h < j + pool_size ;++h){
                    result[i][j] = fmaxf(input[k][h],result[i][j]);
                }
            }
        }
    }

    return result;
}

vector<vector<float> > avg_pooling(vector<vector<float> >& input,int pool_size)
{
    int n=sz(input);
    int output_size = n-pool_size+1;

    vector<vector<float> > result(output_size,vector<float>(output_size,0.0));

    for(int i=0;i<output_size;i++)
    {
        for(int j=0;j<output_size;j++)
        {
            result[i][j]=0.0;
            for(int k=0;k<pool_size;k++)
            {
                for(int l=0;l<pool_size;l++)
                {
                    result[i][j] += input[i+k][j+l];
                }
            }

            result[i][j]/=(pool_size * pool_size);
        }
    }

    return result;
}

vector<float> output_probability(vector<float> &input_vector,string type)
{
    int n=sz(input_vector);
    vector<float> output(n);
    if(type == "softmax")
    {
        float total = 0.0;
        for(int i=0;i<n;++i) total+=(float)exp(input_vector[i]);
        for(int i=0;i<n;++i) output[i] = exp(input_vector[i])/total;
    }
    else if(type == "sigmoid")
    {
        for(int i=0;i<n;++i) output[i] = 1/(1+exp(-input_vector[i]));
    }
    return output;
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
        int output_size = n + 2*p - m + 1;
        vector<vector<float> > input(n, vector<float>(n));
        vector<vector<float> > kernel(m, vector<float>(m));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                ss >> input[i][j];
            }
        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                ss >> kernel[i][j];
            }
        }
        vector<vector<float> > output = convolve(input, kernel, p);
        
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                cout << output[i][j] << " ";
            }
            cout << endl;
        }
    }
    else if (work == 2){
        int activation_type;
        ss >> activation_type;
        int n, m;
        ss >> n >> m;
        vector<vector<float> > input(n, vector<float>(m));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                ss >> input[i][j];
            }
        }
        vector<vector<float> > output(n, vector<float>(m));
        if(activation_type == 0)
            output = activate(input, "relu");
        else if(activation_type == 1)
            output = activate(input, "tanh");

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                cout << output[i][j] << " ";
            }
            cout << endl;
        }
    }
    else if (work == 3){
        int pool_type;
        ss >> pool_type;

        int pool_size;
        ss >> pool_size;

        int n;
        ss >> n;

        // pool_size = 2;
        
        vector<vector<float> > input(n, vector<float>(n));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                ss >> input[i][j];
            }
        }

        int o_n = n - pool_size + 1;

        vector<vector<float> > output(o_n, vector<float>(o_n));

        if(pool_type == 0)
            output = max_pooling(input, pool_size);
        else if(pool_type == 1)
            output = avg_pooling(input, pool_size);

        for (int i = 0; i < o_n; ++i) {
            for (int j = 0; j < o_n; ++j) {
                cout << output[i][j] << " ";
            }
            cout << endl;
        }
    }
    else if (work == 4){
        int normalization_type;
        ss >> normalization_type;
        
        vector<float> input(argc-3);
        for (int i = 0; i < argc-3; ++i) {
            ss >> input[i];
        }
        vector<float> output;

        if(normalization_type == 0)
            output = output_probability(input, "sigmoid");
        else if(normalization_type == 1)
            output = output_probability(input, "softmax");

        for (int i = 0; i < argc-3; ++i) {
            cout << output[i] << " ";
        }
        cout << endl;

    }
}