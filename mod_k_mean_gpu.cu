#include<iostream>
#include<string>
#include<malloc.h>
#include<fstream>
#include<sstream>
#include<vector>
#include<cmath>
#include<cstdio>
#include<stdlib.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include <map>
#include <iomanip>
#include <sys/time.h>
#include<assert.h>
#define THREADSPERBLOCK 256
#define EPS 0.01  

using namespace std;

template <class T>
__device__ static T distanceComponentGPU(T *elementA, T *elementB)
{
    T dist = 0.0f;
    dist = elementA[0] - elementB[0];
    dist = dist * dist;
    return dist; 
}

template <class T>
__device__ static T distanceFinalizeGPU(int n_dim, T *components)
{
    T dist = 0.0f;
    for (unsigned int cnt = 0; cnt < n_dim; cnt++) dist += components[cnt];
    dist = sqrt(dist);
    return dist;
}

template <class T>
__device__ static T distanceGPU(int n_dim, T *elementA, T *elementB) 
{
    T dist = 0.0f;
    for (unsigned int cnt = 0; cnt < n_dim; cnt++)
    {
        T di = (elementA[cnt] - elementB[cnt]);
        dist += di * di;
    }
    dist = sqrt(dist);
    return dist; 
    
}

class Internal;

class DataIO
{
public:
    DataIO();
    ~DataIO();
    float* readData(const char* fileName);
    float* getData(); 
    const char* getFileName();
    int getNumElements(); 
    int getNumClusters(); 
    int getDimensions();
    int getDataSize(); 
    void setDataSize(int numData); 
    void printClusters(int numData, int numClust, int numDim, float *data, float *ctr, int *assign); 
    
    template <typename T>
    T allocZeroedDeviceMemory(int memSize)
    {
        T retVal;
        cudaMalloc((void**) &retVal, memSize);
        cudaMemset(retVal, 0, memSize);
        return retVal;
    }
    
    template <typename T>
    T allocInitializedDeviceMemory(int memSize, int preSet)
    {
        T retVal;
        cudaMalloc((void**) &retVal, memSize);
        cudaMemset(retVal, preSet, memSize);
        return retVal;
    }
    
    template <typename T>
    T allocDeviceMemory(int memSize, T data)
    {
        T retVal;
        cudaMalloc((void**) &retVal, memSize);
        cudaMemcpy(retVal, data, memSize, cudaMemcpyHostToDevice); 
        return retVal;
    }
   
    template <typename T>
    T allocDeviceMemory(int memSize)
    {
        T retVal;
        cudaMalloc((void**) &retVal, memSize);
        return retVal;
    }
    
private:
    Internal* ip;
};

class Internal 
{
private:
    int N; 
    int K; 
    int n_dim; 
    int dataSize; 
    bool deviceCheck; 
    bool printTime; 
    float* data;
    const char* fileName; 
    const char* execName; 
    
public:
    
    Internal()
    {
        N=K= dataSize = 0;
	n_dim=0;
        deviceCheck = true;
        printTime = true;
        data = NULL;
    }
  
    ~ Internal()
    {
        delete data;
    }

    int getNumElements() { return N; };
    
    int getNumClusters() { return K; };
    
    int getDimensions() { return n_dim; };
    
    const char* getExecName()
    {
        return execName;
    }
    
    const char* getFileName()
    {
        return fileName;
    }
    
    int getDataSize()
    {
        return dataSize;
    }
    
    void setExecName(const char* en)
    {
        execName = en;
    }
    
    void setFileName(const char* fn)
    {
        fileName = fn;
    }
    
    void setDataSize(int numData)
    {
        dataSize = numData;
    }
    
    float* getData()
    {
        return data;
    }  
    
    void printParams()
    {
        cout<<"Number of Conformations : "<<N<<endl;
	cout<<"Number of Clusters : "<<K<<endl;
    }
    
    float* readFile(const char* fileName)
    {
        
        string line;
        ifstream infile;
        float pars[3];
        int numData;
        infile.open(fileName, ios::in);
        if (!infile.is_open())
        {
            cout << "Error in readFile(): Unable to find or open file \"" << fileName << "\"." << endl;
            exit(1);
        }
        assert(!infile.fail());
        try
        {
            for (int i = 0; i < 3; i++)
            {
                getline(infile, line);
                if (infile.eof()) throw 42;
                istringstream buffer(line);
                if (!(buffer >> pars[i])) throw 1337;
            }
            N = (int) pars[0]; 
            K = (int) pars[1]; 
            n_dim = (int) pars[2]; 
            if ((numData = dataSize) == 0) 
            {
                printParams();
                numData = N * n_dim;
            }
            data = (float*) malloc(sizeof(float) * numData);
            memset(data, 0, sizeof(float) * numData);
            for (int i = 0; i < numData; i++) 
            {
                getline(infile, line);
                if (infile.eof()) throw 42;
                istringstream buffer(line);
                if (!(buffer >> data[i])) throw 1337;
            }
        }
        catch (int e)
        {
            cout << "Error in dataIO::readFile(): ";
            if (e == 42) cout << "reached end of file \"" << fileName << "\" prematurely" << endl;
            else if (e == 1337) cout << "can only read floating point numbers" << endl;
            else cout << "reading file content failed" << endl;
            cout << "                             Please check parameters and file format" << endl;
            return NULL;
        }
        infile.close();
        assert(!infile.fail());
        return data;
    }
    
}; 

DataIO::DataIO()
{
    ip = new Internal;
}

DataIO::~DataIO()
{
    delete ip;
}

float* DataIO::readData(const char* fileName)
{  
    float* data;
    data = ip->readFile(fileName);
    return data;
}

float* DataIO::getData()
{
    return ip->getData();
}

const char* DataIO::getFileName()
{
    return ip->getFileName();
}

int DataIO::getNumElements() { return ip->getNumElements(); }

int DataIO::getNumClusters() { return ip->getNumClusters(); }

int DataIO::getDimensions() { return ip->getDimensions(); }

int DataIO::getDataSize() { return ip->getDataSize(); }

void DataIO::printClusters(int numData, int numClust, int n_dim, float *data, float *ctr, int *assign)
{
    cout << "Data clusters:" << endl;
    for (int i = 0; i < numClust; i++)
        {
            cout << "Cluster " << i << " (";
            int count = 0;
            for (int j = 0; j < numData; j++)
            {
                if (assign[j] == i) 
                {
                    // print out vectors
                    cout << "{";
                    for (int cnt = 0; cnt < n_dim; cnt++) cout << data[n_dim * j + cnt] << ((cnt < n_dim-1) ? ", " : "");
                    cout << "}, ";
                    count++;
                }
            }
            if (count > 0) cout << "\b\b";
            if (ctr != NULL) 
            {
                cout << ") ctr {";
                for (int cnt = 0; cnt < n_dim; cnt++) cout << ctr[n_dim * i + cnt] << ", ";
                cout << "\b\b}" << endl;
            }
            else cout << ")" << endl;
        }
}


class Timing{
    
    
private:
    map<string, cudaEvent_t> startMap; 
    map<string, cudaEvent_t> stopMap; 
    
public:
    Timing();
    ~Timing();
    
    void start(string timerName);
    void stop(string timerName);
    void report();
    void report(string timerName);
};

Timing::Timing(){
}

Timing::~Timing(){
}

void Timing::start(string timerName){
    cudaEventCreate(&startMap[timerName]);
    cudaEventRecord(startMap[timerName], 0);
}


void Timing::stop(string timerName){
    cudaEventCreate(&stopMap[timerName]);  
    cudaEventRecord(stopMap[timerName], 0);
}


void Timing::report(){
    cudaEvent_t currentTime;
    cudaEventCreate(&currentTime);
    cudaEventRecord(currentTime,0);
    float timeMs;
    string status = "";
    
    cout << "Current Timings:" << endl;
    cout << setw(15) << "Timer" << setw(15) <<  "Time (ms)" << setw(15) << "Status" << endl;
    for( map<string, cudaEvent_t>::iterator it=startMap.begin(); it!=startMap.end() ; ++it){
        if(stopMap.find((*it).first) != stopMap.end()){
            cudaEventElapsedTime(&timeMs, (*it).second, stopMap[(*it).first]);
            status="done";
        } else {
            cudaEventElapsedTime(&timeMs, (*it).second , currentTime);
            status="running";
        }
        
        cout << setw(15) << (*it).first << setw(15) << timeMs << setw(15) << status << endl;
    }
}

void Timing::report(string timerName){
    cudaEvent_t currentTime;
    cudaEventCreate(&currentTime);
    cudaEventRecord(currentTime,0);
    float timeMs;
    
    if(startMap.find(timerName) == startMap.end()){
        cout << "Timer \"" << timerName << "\" was never started." << endl;
        return;
    } else if(stopMap.find(timerName) == stopMap.end()){
        cudaEventElapsedTime(&timeMs, startMap[timerName], currentTime);
        cout << timerName << " = " << timeMs << " ms (running)" << endl;
        return;
    }
    cudaEventElapsedTime(&timeMs, startMap[timerName], stopMap[timerName]);
    cout << timerName << " = " << timeMs << " ms" << endl;
}


template <unsigned int BLOCKSIZE, class T>
__device__ static void reduceOne(int tid, T *s_A)
{
    if (BLOCKSIZE >= 1024) { if (tid < 512) { s_A[tid] += s_A[tid + 512]; } __syncthreads(); }
    if (BLOCKSIZE >=  512) { if (tid < 256) { s_A[tid] += s_A[tid + 256]; } __syncthreads(); }
    if (BLOCKSIZE >=  256) { if (tid < 128) { s_A[tid] += s_A[tid + 128]; } __syncthreads(); }
    if (BLOCKSIZE >=  128) { if (tid <  64) { s_A[tid] += s_A[tid +  64]; } __syncthreads(); }
    
    if (tid < 32)
    {
        if (BLOCKSIZE >= 64) { s_A[tid] += s_A[tid + 32]; }
        if (BLOCKSIZE >= 32) { s_A[tid] += s_A[tid + 16]; }
        if (BLOCKSIZE >= 16) { s_A[tid] += s_A[tid +  8]; }
        if (BLOCKSIZE >=  8) { s_A[tid] += s_A[tid +  4]; }
        if (BLOCKSIZE >=  4) { s_A[tid] += s_A[tid +  2]; }
        if (BLOCKSIZE >=  2) { s_A[tid] += s_A[tid +  1]; }
    }
}


template <unsigned int BLOCKSIZE, class T, class U>
__device__ static void reduceTwo(int tid, T *s_A, U *s_B)
{
    if (BLOCKSIZE >= 1024) { if (tid < 512) { s_A[tid] += s_A[tid + 512]; s_B[tid] += s_B[tid + 512]; } __syncthreads(); }
    if (BLOCKSIZE >=  512) { if (tid < 256) { s_A[tid] += s_A[tid + 256]; s_B[tid] += s_B[tid + 256]; } __syncthreads(); }
    if (BLOCKSIZE >=  256) { if (tid < 128) { s_A[tid] += s_A[tid + 128]; s_B[tid] += s_B[tid + 128]; } __syncthreads(); }
    if (BLOCKSIZE >=  128) { if (tid <  64) { s_A[tid] += s_A[tid +  64]; s_B[tid] += s_B[tid +  64]; } __syncthreads(); }
    
    if (tid < 32)
    {
        if (BLOCKSIZE >= 64) { s_A[tid] += s_A[tid + 32]; s_B[tid] += s_B[tid + 32]; }
        if (BLOCKSIZE >= 32) { s_A[tid] += s_A[tid + 16]; s_B[tid] += s_B[tid + 16]; }
        if (BLOCKSIZE >= 16) { s_A[tid] += s_A[tid +  8]; s_B[tid] += s_B[tid +  8]; }
        if (BLOCKSIZE >=  8) { s_A[tid] += s_A[tid +  4]; s_B[tid] += s_B[tid +  4]; }
        if (BLOCKSIZE >=  4) { s_A[tid] += s_A[tid +  2]; s_B[tid] += s_B[tid +  2]; }
        if (BLOCKSIZE >=  2) { s_A[tid] += s_A[tid +  1]; s_B[tid] += s_B[tid +  1]; }
    }
}


__global__ static void assignToClusters_KMCUDA(int N, int K, int n_dim, float *X, float *CTR, int *ASSIGN)
{
    extern __shared__ float array[];                        
    float *s_center = (float*) array;                       
    
    unsigned int t = blockDim.x * blockIdx.x + threadIdx.x; 
    unsigned int tid = threadIdx.x;                         
    
    if (t < N)
    {
        float minDist  = 0.0;
        int   minIndex = 0;
        for (unsigned int k = 0; k < K; k++)
        {
            float dist = 0.0;
            unsigned int offsetD = 0;
            while (offsetD < n_dim)
            {
                if (offsetD + tid < n_dim) s_center[tid] = CTR[k * n_dim + offsetD + tid];
                __syncthreads();
                for (unsigned int d = offsetD; d < min(offsetD + blockDim.x, n_dim); d++)
                {
                    dist += distanceComponentGPU(s_center + (d - offsetD), X + (d * N + t));
                }
                offsetD += blockDim.x;
                __syncthreads();
            }
            dist = distanceFinalizeGPU<float>(1, &dist);
            if (dist < minDist || k == 0)
            {
                minDist = dist;
                minIndex = k;
            }
        }
        ASSIGN[t] = minIndex;
    }
}


__global__ static void calcScore_CUDA(int N, int n_dim, float *X, float *CTR, int *ASSIGN, float *SCORE)
{
    extern __shared__ float array[];                     
    float *s_scores = (float*) array;                    
    float *s_center = (float*) &s_scores[blockDim.x];    
    
    int k   = blockIdx.x;                                
    int tid = threadIdx.x;                              
    s_scores[tid] = 0.0;
    unsigned int offsetN = tid;
    while (offsetN < N)
    {
        float dist = 0.0;
        unsigned int offsetD = 0;
        while (offsetD < n_dim)
        {
            if (offsetD + tid < n_dim) s_center[tid] = CTR[k * n_dim + offsetD + tid];
            __syncthreads();
            if (ASSIGN[offsetN] == k)
            {
                for (unsigned int d = offsetD; d < min(offsetD + blockDim.x, n_dim); d++)
                {
                    dist += distanceComponentGPU(s_center + (d - offsetD), X + (d * N + offsetN));
                }
            }
            offsetD += blockDim.x;
            __syncthreads();
        }
        s_scores[tid] += distanceFinalizeGPU(1, &dist);
        offsetN += blockDim.x;
    }
    __syncthreads();
    reduceOne<THREADSPERBLOCK>(tid, s_scores);
    if (tid == 0) SCORE[k] = s_scores[tid];
}



__global__ static void calcCentroids_CUDA(int N, int n_dim, float *X, float *CTR, int *ASSIGN)
{
    extern __shared__ float array[];                            
    int   *s_numElements = (int*)   array;                      
    float *s_centerParts = (float*) &s_numElements[blockDim.x]; 
    
    int k   = blockIdx.x;                                       
    int tid = threadIdx.x;                                      
    
    float clusterSize = 0.0;                                    
    s_numElements[tid] = 0;
    for (unsigned int d = 0; d < n_dim; d++)
    {
        s_centerParts[tid] = 0.0;
        unsigned int offset = tid;
        while (offset < N)
        {
            if (ASSIGN[offset] == k)
            {
                s_centerParts[tid] += X[d * N + offset];
                if (d == 0) s_numElements[tid]++;
            }
            offset += blockDim.x;
        }
        __syncthreads();
        
        if (d == 0)
        {
            reduceTwo<THREADSPERBLOCK>(tid, s_centerParts, s_numElements);
            if (tid == 0) clusterSize = (float) s_numElements[tid];
        }
        else
        {
	    reduceOne<THREADSPERBLOCK>(tid, s_centerParts);
        }
        if (tid == 0) if (clusterSize > 0) CTR[k * n_dim + d] = s_centerParts[tid] / clusterSize;
    }
}


float kmeansGPU(int N, int K, int n_dim, float *x, float *ctr, int *assign, unsigned int maxIter, DataIO *data)
{
    dim3 block(THREADSPERBLOCK);
    dim3 gridK(K);
    dim3 gridN((int)ceil((float)N/(float)THREADSPERBLOCK));
    int sMemAssign=(sizeof(float)*THREADSPERBLOCK);
    int sMemScore=(sizeof(float)*2*THREADSPERBLOCK);
    int sMemCenters=(sizeof(float)*THREADSPERBLOCK+sizeof(int)*THREADSPERBLOCK);
    float *x_d = data->allocDeviceMemory<float*>(sizeof(float) * N * n_dim, x);
    float *ctr_d = data->allocDeviceMemory<float*>(sizeof(float) * K * n_dim, ctr);
    int *assign_d = data->allocDeviceMemory<int*>(sizeof(int) * N);
    float *s_d = data->allocZeroedDeviceMemory<float*>(sizeof(float) * K);
    float *s   = (float*) malloc(sizeof(float) * K);
    float oldscore = -1000.0, score = 0.0;
    if (maxIter < 1) maxIter = INT_MAX;
    unsigned int iter = 0;
    while (iter < maxIter && ((score - oldscore) * (score - oldscore)) > EPS)
    {
        oldscore = score;
        if (iter > 0)
        {
            calcCentroids_CUDA<<<gridK, block, sMemCenters>>>(N, n_dim, x_d, ctr_d, assign_d);
        }
        iter++;
        assignToClusters_KMCUDA<<<gridN, block, sMemAssign>>>(N, K, n_dim, x_d, ctr_d, assign_d);
        calcScore_CUDA<<<gridK, block, sMemScore>>>(N, n_dim, x_d, ctr_d, assign_d, s_d);
        cudaMemcpy(s, s_d,    sizeof(float) * K, cudaMemcpyDeviceToHost);
        score = 0.0;
        for (int i = 0; i < K; i++) score += s[i];
    }
    cout << "Number of iterations: " << iter << endl;
    cudaMemcpy(ctr, ctr_d,         sizeof(float) * K * n_dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(assign, assign_d,    sizeof(int)   * N    , cudaMemcpyDeviceToHost);
    cudaFree(x_d);
    cudaFree(ctr_d);
    cudaFree(assign_d);
    cudaFree(s_d);
    free(s);
    return score;
}

int main()
{
    Timing timer;
    cudaSetDevice(0);
    DataIO* data = new DataIO;
    float score = 0.0f;
    float* x = data->readData("alanine_2000MB.dat");
    int N = data->getNumElements();
    int K = data->getNumClusters();
    int n_dim = data->getDimensions();
    float* ctr = (float*) malloc(sizeof(float) * K * n_dim);
    memset(ctr, 0, sizeof(float) * K * n_dim);
    int* assign = (int*) malloc(sizeof(int) * N);
    memset(assign, 0, sizeof(int) * N);
    for (unsigned int k = 0; k < K; k++)
    {
        for (unsigned int d = 0; d < n_dim; d++)
        {
             ctr[k * n_dim + d] = x[d * N + k];
        }
    }
    timer.start("kmeansGPU");
    score = kmeansGPU(N, K, n_dim, x, ctr, assign, (unsigned int)0, data);
    timer.stop("kmeansGPU");
    // data->printClusters(N, K, D, x, ctr, assign);  
    timer.report();
    free(x);
    free(ctr);
    free(assign);
    cout << "Done clustering" << endl;
    return 0;
}
