#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py=pybind11;
namespace fs=std::filesystem;

// Multi-dimensional tensor with support for gradients
class Tensor{
public:
    std::vector<float> data;
    std::vector<float> grad;
    std::vector<float> velocity; 
    std::vector<int> shape;
    bool requires_grad;

    Tensor(std::vector<int> s,bool req_grad=false):shape(s),requires_grad(req_grad){
        int size=1;
        for(int dim:shape)size*=dim;
        data.resize(size,0.0f);
        if(requires_grad)grad.resize(size,0.0f);
    }

    // Access element in 4D tensor (batch, channel, height, width)
    float& at(int n,int c,int h,int w){
        int idx=n*(shape[1]*shape[2]*shape[3])+c*(shape[2]*shape[3])+h*shape[3]+w;
        return data[idx];
    }
    const float& at(int n,int c,int h,int w)const{
        int idx=n*(shape[1]*shape[2]*shape[3])+c*(shape[2]*shape[3])+h*shape[3]+w;
        return data[idx];
    }
    // Access element in 2D tensor (batch, feature)
    float& at(int n,int d){return data[n*shape[1]+d];}
    const float& at(int n,int d)const{return data[n*shape[1]+d];}

    void zero_grad(){std::fill(grad.begin(),grad.end(),0.0f);}

    // Initialize weights using Xavier uniform distribution
    void xavier_init(){
        std::default_random_engine generator;
        float limit=sqrt(6.0f/(float)data.size());
        std::uniform_real_distribution<float> distribution(-limit,limit);
        for(auto& val:data)val=distribution(generator);
    }

    // Save tensor to binary file
    void save(const std::string& path){
        std::ofstream out(path,std::ios::binary);
        int ndim=shape.size();
        out.write(reinterpret_cast<const char*>(&ndim),sizeof(int));
        out.write(reinterpret_cast<const char*>(shape.data()),ndim*sizeof(int));
        out.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(float));
        out.close();
    }

    // Load tensor from binary file
    void load(const std::string& path){
        std::ifstream in(path,std::ios::binary);
        int ndim;
        in.read(reinterpret_cast<char*>(&ndim),sizeof(int));
        std::vector<int> loaded_shape(ndim);
        in.read(reinterpret_cast<char*>(loaded_shape.data()),ndim*sizeof(int));
        int loaded_size=1;
        for(int s:loaded_shape)loaded_size*=s;
        this->shape=loaded_shape;
        this->data.resize(loaded_size);
        if(requires_grad)this->grad.resize(loaded_size,0.0f);
        in.read(reinterpret_cast<char*>(data.data()),data.size()*sizeof(float));
        in.close();
    }
};

// Holds images, labels, and class names for a dataset
struct Dataset{
    std::vector<Tensor> images;
    std::vector<int> labels;
    std::vector<std::string> class_names;
};

// Load images from directory, resize to 32x32, normalize to [0,1]
Dataset load_dataset_cpp(const std::string& path){
    Dataset ds;
    int label_idx=0;
    if(!fs::exists(path))throw std::runtime_error("Path not found: "+path);
    for(const auto& entry:fs::directory_iterator(path)){
        if(entry.is_directory()){
            ds.class_names.push_back(entry.path().filename().string());
            for(const auto& img_entry:fs::directory_iterator(entry.path())){
                cv::Mat img=cv::imread(img_entry.path().string(),cv::IMREAD_COLOR);
                if(img.empty())continue;
                cv::resize(img,img,cv::Size(32,32));
                Tensor t({1,3,32,32},false);
                for(int h=0;h<32;++h){
                    for(int w=0;w<32;++w){
                        cv::Vec3b pixel=img.at<cv::Vec3b>(h,w);
                        t.at(0,0,h,w)=pixel[2]/255.0f;
                        t.at(0,1,h,w)=pixel[1]/255.0f;
                        t.at(0,2,h,w)=pixel[0]/255.0f;
                    }
                }
                ds.images.push_back(t);
                ds.labels.push_back(label_idx);
            }
            label_idx++;
        }
    }
    return ds;
}

// 2D convolution layer
class Conv2D{
public:
    int in_c,out_c,k_size,stride,padding;
    Tensor weights,bias,input_cache;
    long long macs=0,params=0;
    Conv2D(int ic,int oc,int k,int s,int p)
        :in_c(ic),out_c(oc),k_size(k),stride(s),padding(p),
         weights({oc,ic,k,k},true),bias({oc},true),input_cache({0}){
        weights.xavier_init();
        params=weights.data.size()+bias.data.size();
    }
    // Forward pass: apply convolution filters on input
    Tensor forward(const Tensor& input){
        input_cache=input;
        int N=input.shape[0];
        int H_out=(input.shape[2]+2*padding-k_size)/stride+1;
        int W_out=(input.shape[3]+2*padding-k_size)/stride+1;
        Tensor output({N,out_c,H_out,W_out},true);
        macs=(long long)N*out_c*H_out*W_out*(in_c*k_size*k_size);
        for(int n=0;n<N;++n){
            for(int oc=0;oc<out_c;++oc){
                for(int h=0;h<H_out;++h){
                    for(int w=0;w<W_out;++w){
                        float sum=bias.data[oc];
                        int h_start=h*stride-padding;
                        int w_start=w*stride-padding;
                        for(int ic=0;ic<in_c;++ic){
                            for(int kh=0;kh<k_size;++kh){
                                for(int kw=0;kw<k_size;++kw){
                                    int h_in=h_start+kh;
                                    int w_in=w_start+kw;
                                    if(h_in>=0&&h_in<input.shape[2]&&w_in>=0&&w_in<input.shape[3])
                                        sum+=input.at(n,ic,h_in,w_in)*weights.at(oc,ic,kh,kw);
                                }
                            }
                        }
                        output.at(n,oc,h,w)=sum;
                    }
                }
            }
        }
        return output;
    }
    // Backward pass: compute gradients for weights, bias, and input
    Tensor backward(const Tensor& grad_output){
        Tensor grad_input(input_cache.shape,false);
        int N=input_cache.shape[0];
        int H_out=grad_output.shape[2];
        int W_out=grad_output.shape[3];
        for(int n=0;n<N;++n){
            for(int oc=0;oc<out_c;++oc){
                for(int h=0;h<H_out;++h){
                    for(int w=0;w<W_out;++w){
                        float g=grad_output.at(n,oc,h,w);
                        bias.grad[oc]+=g;
                        int h_start=h*stride-padding;
                        int w_start=w*stride-padding;
                        for(int ic=0;ic<in_c;++ic){
                            for(int kh=0;kh<k_size;++kh){
                                for(int kw=0;kw<k_size;++kw){
                                    int h_in=h_start+kh;
                                    int w_in=w_start+kw;
                                    if(h_in>=0&&h_in<input_cache.shape[2]&&w_in>=0&&w_in<input_cache.shape[3]){
                                        weights.grad[oc*(in_c*k_size*k_size)+ic*(k_size*k_size)+kh*k_size+kw]+=
                                            input_cache.at(n,ic,h_in,w_in)*g;
                                        grad_input.at(n,ic,h_in,w_in)+=weights.at(oc,ic,kh,kw)*g;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return grad_input;
    }
};

// 2D max pooling layer
class MaxPool2D{
    int k_size,stride;
    Tensor input_cache;
    std::vector<int> max_indices;
public:
    long long macs=0,params=0;
    MaxPool2D(int k,int s):k_size(k),stride(s),input_cache({0}){}
    Tensor forward(const Tensor& input){
        input_cache=input;
        int N=input.shape[0];int C=input.shape[1];
        int H_out=(input.shape[2]-k_size)/stride+1;
        int W_out=(input.shape[3]-k_size)/stride+1;
        Tensor output({N,C,H_out,W_out},true);
        max_indices.assign(output.data.size(),-1);
        for(int n=0;n<N;++n){
            for(int c=0;c<C;++c){
                for(int h=0;h<H_out;++h){
                    for(int w=0;w<W_out;++w){
                        float max_val=-1e9;
                        int max_idx=-1;
                        for(int kh=0;kh<k_size;++kh){
                            for(int kw=0;kw<k_size;++kw){
                                int h_in=h*stride+kh;
                                int w_in=w*stride+kw;
                                int idx=n*(C*input.shape[2]*input.shape[3])+c*(input.shape[2]*input.shape[3])+h_in*input.shape[3]+w_in;
                                if(input.data[idx]>max_val){max_val=input.data[idx];max_idx=idx;}
                            }
                        }
                        int out_idx=n*(C*H_out*W_out)+c*(H_out*W_out)+h*W_out+w;
                        output.data[out_idx]=max_val;
                        max_indices[out_idx]=max_idx;
                    }
                }
            }
        }
        return output;
    }
    Tensor backward(const Tensor& grad_output){
        Tensor grad_input(input_cache.shape,false);
        for(size_t i=0;i<grad_output.data.size();++i){
            if(max_indices[i]!=-1)grad_input.data[max_indices[i]]+=grad_output.data[i];
        }
        return grad_input;
    }
};

// ReLU activation: max(0, x)
class ReLU{
    Tensor input_cache;
public:
    long long macs=0,params=0;
    ReLU():input_cache({0}){}
    Tensor forward(const Tensor& input){
        input_cache=input;
        Tensor out=input;
        for(auto& val:out.data)val=std::max(0.0f,val);
        return out;
    }
    Tensor backward(const Tensor& grad_output){
        Tensor grad_input=grad_output;
        for(size_t i=0;i<grad_input.data.size();++i)
            if(input_cache.data[i]<=0)grad_input.data[i]=0;
        return grad_input;
    }
};

// Fully connected layer
class Linear{
public:
    int in_f,out_f;
    Tensor weights,bias,input_cache;
    long long macs=0,params=0;
    Linear(int in,int out):in_f(in),out_f(out),
        weights({in,out},true),bias({out},true),input_cache({0}){
        weights.xavier_init();
        params=weights.data.size()+bias.data.size();
    }
    Tensor forward(const Tensor& input){
        input_cache=input;
        int N=input.shape[0];
        Tensor output({N,out_f},true);
        macs=(long long)N*in_f*out_f;
        for(int n=0;n<N;++n){
            for(int out=0;out<out_f;++out){
                float sum=bias.data[out];
                for(int in=0;in<in_f;++in)sum+=input.at(n,in)*weights.at(in,out);
                output.at(n,out)=sum;
            }
        }
        return output;
    }
    Tensor backward(const Tensor& grad_output){
        Tensor grad_input(input_cache.shape,false);
        int N=input_cache.shape[0];
        for(int n=0;n<N;++n){
            for(int out=0;out<out_f;++out){
                float g=grad_output.at(n,out);
                bias.grad[out]+=g;
                for(int in=0;in<in_f;++in){
                    weights.grad[in*out_f+out]+=input_cache.at(n,in)*g;
                    grad_input.at(n,in)+=weights.at(in,out)*g;
                }
            }
        }
        return grad_input;
    }
};

// Flatten multi-dimensional input into 2D (batch, features)
class Flatten{
    std::vector<int> input_shape;
public:
    long long macs=0,params=0;
    Tensor forward(const Tensor& input){
        input_shape=input.shape;
        int N=input.shape[0];
        int size=1;
        for(size_t i=1;i<input.shape.size();++i)size*=input.shape[i];
        Tensor out=input;
        out.shape={N,size};
        return out;
    }
    Tensor backward(const Tensor& grad_output){
        Tensor out=grad_output;
        out.shape=input_shape;
        return out;
    }
};

// Combine multiple tensors into a single batched tensor
Tensor batch_tensors(const std::vector<Tensor>& batch){
    if(batch.empty())return Tensor({0});
    int N=batch.size();
    auto s=batch[0].shape;s[0]=N;
    Tensor out(s);
    size_t sz=batch[0].data.size();
    for(int i=0;i<N;++i)std::copy(batch[i].data.begin(),batch[i].data.end(),out.data.begin()+i*sz);
    return out;
}

// Compute cross-entropy loss and its gradient
float cross_entropy_loss(const Tensor& logits, const std::vector<int>& targets, Tensor& grad_input) {
    int N = logits.shape[0];
    int C = logits.shape[1];
    grad_input = logits; grad_input.zero_grad();
    float loss = 0;

    for (int n = 0; n < N; ++n) {
        // Subtract max for numerical stability
        float max_l = -1e9;
        for (int c = 0; c < C; ++c) max_l = std::max(max_l, logits.data[n * C + c]);
        float sum = 0;
        std::vector<float> exps(C);
        // Compute softmax probabilities
        for (int c = 0; c < C; ++c) { 
            exps[c] = std::exp(logits.data[n * C + c] - max_l); 
            sum += exps[c]; 
        }
        
        float prob = exps[targets[n]] / sum;
        loss += -std::log(prob + 1e-7f); // epsilon to prevent log(0) 

        for (int c = 0; c < C; ++c) {
            float s = exps[c] / sum;
            grad_input.data[n * C + c] = (c == targets[n] ? s - 1 : s) / N;
        }
    }
    return loss / N;
}

// SGD optimizer with momentum
class SGD {
    float lr;
    float momentum;
public:
    SGD(float l, float m=0.9f) : lr(l), momentum(m) {}

    // Update weights and biases using momentum
    void step(Tensor& w, Tensor& b) {
        if (w.velocity.empty()) w.velocity.resize(w.data.size(), 0.0f);
        for (size_t i = 0; i < w.data.size(); ++i) {
            w.velocity[i] = momentum * w.velocity[i] - lr * w.grad[i];
            w.data[i] += w.velocity[i];
            w.grad[i] = 0;
        }
        if (b.velocity.empty()) b.velocity.resize(b.data.size(), 0.0f);
        for (size_t i = 0; i < b.data.size(); ++i) {
            b.velocity[i] = momentum * b.velocity[i] - lr * b.grad[i];
            b.data[i] += b.velocity[i];
            b.grad[i] = 0;
        }
    }
};

// Expose classes and functions to Python via pybind11
PYBIND11_MODULE(my_dl_framework,m){
    py::class_<Tensor>(m,"Tensor")
        .def(py::init<std::vector<int>,bool>(),py::arg("shape"),py::arg("requires_grad")=false)
        .def_readwrite("data",&Tensor::data)
        .def_readwrite("grad",&Tensor::grad)
        .def_readwrite("shape",&Tensor::shape)
        .def("save",&Tensor::save)
        .def("load",&Tensor::load);

    py::class_<Dataset>(m,"Dataset")
        .def_readwrite("images",&Dataset::images)
        .def_readwrite("labels",&Dataset::labels)
        .def_readwrite("class_names",&Dataset::class_names);

    m.def("load_dataset",&load_dataset_cpp);
    m.def("batch_tensors",&batch_tensors);
    m.def("cross_entropy_loss",&cross_entropy_loss);

    py::class_<Conv2D>(m,"Conv2D")
        .def(py::init<int,int,int,int,int>())
        .def("forward",&Conv2D::forward)
        .def("backward",&Conv2D::backward)
        .def_readwrite("weights",&Conv2D::weights)
        .def_readwrite("bias",&Conv2D::bias)
        .def_readwrite("macs",&Conv2D::macs)
        .def_readwrite("params",&Conv2D::params);

    py::class_<MaxPool2D>(m,"MaxPool2D")
        .def(py::init<int,int>())
        .def("forward",&MaxPool2D::forward)
        .def("backward",&MaxPool2D::backward)
        .def_readwrite("macs",&MaxPool2D::macs)
        .def_readwrite("params",&MaxPool2D::params);

    py::class_<ReLU>(m,"ReLU")
        .def(py::init<>())
        .def("forward",&ReLU::forward)
        .def("backward",&ReLU::backward);

    py::class_<Flatten>(m,"Flatten")
        .def(py::init<>())
        .def("forward",&Flatten::forward)
        .def("backward",&Flatten::backward);

    py::class_<Linear>(m,"Linear")
        .def(py::init<int,int>())
        .def("forward",&Linear::forward)
        .def("backward",&Linear::backward)
        .def_readwrite("weights",&Linear::weights)
        .def_readwrite("bias",&Linear::bias)
        .def_readwrite("macs",&Linear::macs)
        .def_readwrite("params",&Linear::params);

    py::class_<SGD>(m,"SGD")
        .def(py::init<float, float>(), py::arg("lr"), py::arg("momentum")=0.9f)
        .def("step",&SGD::step);
}