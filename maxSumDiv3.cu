#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cassert>

// maxSumDivThreeCPU()为CPU实现版本，dp每次根据上一个位置的mod0、1、2的最大和更新当前位置mod0、1、2的最大子序列和。
// GPU版本实现思路：分为findMaxSumInSubArrays()和reduceModMaxsum()前后两部分：
// (为了方便实现，这里假设数组总长度能被(numBlocks*blockSize)整除)
// 1. findMaxSumInSubArrays()：首先把整个数组等分给每个block（假设1024个block），每个thread处理totalLength
// (numBlocks*blockSize)个data；每个thread的处理过程和CPU上的dp过程一致，for循环求得这部分数的mod0、1、2的最大子序列和；
// 于是通过findMaxSumInSubArrays()得到了numBlocks*blockSize组数的mod0、1、2的最大子序列和
// 2. reduceModMaxsum()：在每个block内采用reduce归并的方式更新这blockSize组数的mod0、1、2的最大子序列和
// 通过如下的方式对mod3=0，1，2的最大子序列和进行归并：
// 考虑两个数组a,b, 它们mod3=0，1，2的最大子序列和分别为(sumMod0_a,sumMod1_a,sumMod2_a),(sumMod0_b,sumMod1_b,sumMod2_b)
// 则a,b合并的数组c, mod3=0，1，2的最大子序列和为：(特别地，没有对应的mod和时记为负无穷)
// sumMod0_c = max(sumMod0_a+sumMod0_b, sumMod1_a+sumMod2_b, sumMod2_a+sumMod1_b)
// sumMod1_c = max(sumMod0_a+sumMod1_b, sumMod1_a+sumMod0_b, sumMod2_a+sumMod2_b)
// sumMod2_c = max(sumMod0_a+sumMod2_b, sumMod1_a+sumMod1_b, sumMod2_a+sumMod0_b)
// 在modMaxsum数组的第0/1/2个位置即为每个block对应数组mod 3 = 0/1/2的最大子序列和
// 最后在CPU上执行一次循环次数为numBlocks的最大子序列和合并，得到全局的最大子序列和

const long long NEGATIVE_INFINITY = -1e9;
__inline__ __device__ int deviceMax(int a, int b) {
    return a > b? a : b;
}


__global__ void findMaxSumInSubArrays(long long  *input, long long  *modMaxsum, int totalLength, int sub_array_length) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int g_tid = bid*blockDim.x+tid;
    int N = totalLength/gridDim.x; // N：data_preblock
    int n = N/blockDim.x; // n: data_prethread
    long long dp0 = 0, dp1 = 0, dp2 = 0;
    long long cur_dp0 = 0, cur_dp1 = 0, cur_dp2 = 0;

    // 每个thread处理n个data
    for (int i = 0; i < n; i++) {
        // 动态规划更新
        cur_dp0 = dp0;
        cur_dp1 = dp1;
        cur_dp2 = dp2;
        long long num = input[i + g_tid*n];       
        modMaxsum[g_tid*3] += num;
        // bug记录：数组动态分配内存函数提前退出，改为直接分配dp0,1,2
        {
           if((cur_dp0 + num) % 3 == 0){
               dp0 = max(dp0, cur_dp0 + num); 
           }
           else if((cur_dp0 + num) % 3 == 1){
               dp1= max(dp1, cur_dp0 + num); 
           }
           else{
               dp2= max(dp2, cur_dp0 + num); 
           }
        }
        {
           if((cur_dp1 + num) % 3 == 0){
               dp0 = max(dp0, cur_dp1 + num); 
           }
           else if((cur_dp1 + num) % 3 == 1){
               dp1= max(dp1, cur_dp1 + num); 
           }
           else{
               dp2= max(dp2, cur_dp1 + num); 
           }
        }
        {
           if((cur_dp2 + num) % 3 == 0){
               dp0 = max(dp0, cur_dp2 + num); 
           }
           else if((cur_dp2 + num) % 3 == 1){
               dp1= max(dp1, cur_dp2 + num); 
           }
           else{
               dp2= max(dp2, cur_dp2 + num); 
           }
        }
    }
    // modMaxsum记录每个thread统计的子数组的mod0、1、2的最大和
    modMaxsum[g_tid*3] = dp0;
    modMaxsum[g_tid*3+1] = dp1;
    modMaxsum[g_tid*3+2] = dp2;
}


__global__ void reduceModMaxsum(long long  *modMaxsum){
    __shared__ long long sdata[512*3]; //sdata长度为blockSize*3
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 没有对应的mod和时记为负无穷
    sdata[tid*3] = modMaxsum[i*3]>0?modMaxsum[i*3]:NEGATIVE_INFINITY;
    sdata[tid*3+1] = modMaxsum[i*3+1]>0?modMaxsum[i*3+1]:NEGATIVE_INFINITY;
    sdata[tid*3+2] = modMaxsum[i*3+2]>0?modMaxsum[i*3+2]:NEGATIVE_INFINITY;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            long long cur_modsum0 = sdata[tid*3],cur_modsum1 = sdata[tid*3+1],cur_modsum2 = sdata[tid*3+2];
            //printf("%d ", cur_modsum1);
            sdata[tid*3] = deviceMax(deviceMax(cur_modsum0+sdata[(tid+s)*3], cur_modsum1+sdata[(tid+s)*3+2]), cur_modsum2+sdata[(tid+s)*3+1]);
            sdata[tid*3+1] = deviceMax(deviceMax(cur_modsum0+sdata[(tid+s)*3+1], cur_modsum1+sdata[(tid+s)*3]), cur_modsum2+sdata[(tid+s)*3+2]);
            sdata[tid*3+2] = deviceMax(deviceMax(cur_modsum0+sdata[(tid+s)*3+2], cur_modsum2+sdata[(tid+s)*3]), cur_modsum1+sdata[(tid+s)*3+1]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        modMaxsum[blockIdx.x*3] = sdata[0];
        modMaxsum[blockIdx.x*3+1] = sdata[1];
        modMaxsum[blockIdx.x*3+2] = sdata[2];
    }        
}


// CPU实现版本
__host__ long long maxSumDivThreeCPU(long long *input, int length) {
    std::vector<long long > dp(3, 0);
    for (int i=0;i<length;i++) {
        long long num = input[i];
        std::vector<long long> current_dp = dp; 
        for (int sum : current_dp) {
            dp[(sum + num) % 3] = std::max(dp[(sum + num) % 3], sum + num);
        }
    }
    return dp[0]; 
}


int main() {
    int totalLength = 1024*1024*128;
    int numBlocks = 1024;
    int blockSize = 512;
    long long *h_input, *h_modMaxsum, *d_input, *d_modMaxsum;
    h_input = new long long[totalLength];
    h_modMaxsum = new long long[numBlocks*blockSize*3];

    // 随机初始化输入数据
    srand(3);
    for (int i = 0; i < totalLength; i++) {
        h_input[i] = static_cast<long long>(rand()%10+1);
    }

    cudaMalloc((void**)&d_input, totalLength * sizeof(long long));
    cudaMalloc((void**)&d_modMaxsum, numBlocks * blockSize * 3 * sizeof(long long));
    cudaMemcpy(d_input, h_input, totalLength * sizeof(long long), cudaMemcpyHostToDevice);
    assert(totalLength%(numBlocks*blockSize)==0);
    findMaxSumInSubArrays<<<numBlocks, blockSize>>>(d_input, d_modMaxsum, totalLength, totalLength/numBlocks/blockSize);
    // cudaMemcpy(h_modMaxsum, d_modMaxsum, numBlocks * blockSize * 3 * sizeof(int), cudaMemcpyDeviceToHost);
    // for(int i=0;i<10;i++)std::cout<< h_modMaxsum[i] << std::endl;

    // d_modMaxsum 存有numBlocks * blockSize 组{mod0maxsum,mod1maxsum,mod2maxsum}
    // 接下来采用reduce的方法两两合并modmaxsum
    reduceModMaxsum<<<numBlocks, blockSize>>>(d_modMaxsum);

    // 将每个block对应数组的最大和合并，得到全局的最大和
    // 将剩余的存有numBlocks组{mod0maxsum,mod1maxsum,mod2maxsum}循环求出最终的{mod0maxsum,mod1maxsum,mod2maxsum}
    cudaMemcpy(h_modMaxsum, d_modMaxsum, numBlocks * blockSize * 3 * sizeof(long long), cudaMemcpyDeviceToHost);
    std::vector<long long> modmaxsum = std::vector<long long>(3, 0);
    modmaxsum[0] = h_modMaxsum[0]>0?h_modMaxsum[0]:NEGATIVE_INFINITY;
    modmaxsum[1] = h_modMaxsum[1]>0?h_modMaxsum[1]:NEGATIVE_INFINITY;
    modmaxsum[2] = h_modMaxsum[2]>0?h_modMaxsum[2]:NEGATIVE_INFINITY;
    for(int i=1;i<numBlocks;i++){
        long long cur_modsum0 = modmaxsum[0],cur_modsum1 = modmaxsum[1],cur_modsum2 = modmaxsum[2];
        modmaxsum[0] = std::max(std::max(cur_modsum0+h_modMaxsum[i*3], cur_modsum1+h_modMaxsum[i*3+2]), cur_modsum2+h_modMaxsum[i*3+1]);
        modmaxsum[1] = std::max(std::max(cur_modsum0+h_modMaxsum[i*3+1], cur_modsum1+h_modMaxsum[i*3]), cur_modsum2+h_modMaxsum[i*3+2]);
        modmaxsum[2] = std::max(std::max(cur_modsum0+h_modMaxsum[i*3+2], cur_modsum2+h_modMaxsum[i*3]), cur_modsum1+h_modMaxsum[i*3+1]);
    }
    // for(int i=0;i<100;i++)std::cout<< h_modMaxsum[i] << std::endl;

    // 分别输出GPU和CPU实现的结果：
    std::cout << "GPU Result: " << modmaxsum[0]  << std::endl;
    std::cout << "CPU Result: " << maxSumDivThreeCPU(h_input,totalLength)  << std::endl;
    
    
    cudaFree(d_input);
    cudaFree(d_modMaxsum);
    delete[] h_input;
    delete[] h_modMaxsum;
    return 0;
}