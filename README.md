# some_about_aiinfra
## Code 
### pytorch_int8_resnet.py
#### 运行结果
<img width="671" alt="截屏2023-12-08 12 51 33" src="https://github.com/Zhou-Yijie/AI_HPC/assets/118658953/30ea91d1-f09f-4f3e-8c3d-51bdb150ac48">

### maxSumDiv3.cu
#### 实现功能
对数组nums，求能被3整除最大子序列和

#### 实现思路
maxSumDivThreeCPU()为CPU实现版本，dp每次根据上一个位置的mod0、1、2的最大和更新当前位置mod0、1、2的最大子序列和。

GPU版本实现思路：分为findMaxSumInSubArrays()和reduceModMaxsum()前后两部分：
(为了方便实现，这里假设数组总长度能被(numBlocks*blockSize)整除)

1. findMaxSumInSubArrays()：首先把整个数组等分给每个block（假设1024个block），每个thread处理totalLength//(numBlocks*blockSize)个data；每个thread的处理过程和CPU上的dp过程一致，for循环求得这部分数的mod0、1、2的最大子序列和；于是通过findMaxSumInSubArrays()得到了numBlocks*blockSize组数的mod0、1、2的最大子序列和
   
2. reduceModMaxsum()：在每个block内采用reduce归并的方式更新这blockSize组数的mod0、1、2的最大子序列和
通过如下的方式对mod3=0，1，2的最大子序列和进行归并：
考虑两个数组a,b, 它们mod3=0，1，2的最大子序列和分别为(sumMod0_a,sumMod1_a,sumMod2_a),(sumMod0_b,sumMod1_b,sumMod2_b)
则a,b合并的数组c, mod3=0，1，2的最大子序列和为：(特别地，没有对应的mod和时记为负无穷)
```
sumMod0_c = max(sumMod0_a+sumMod0_b, sumMod1_a+sumMod2_b, sumMod2_a+sumMod1_b)
sumMod1_c = max(sumMod0_a+sumMod1_b, sumMod1_a+sumMod0_b, sumMod2_a+sumMod2_b)
sumMod2_c = max(sumMod0_a+sumMod2_b, sumMod1_a+sumMod1_b, sumMod2_a+sumMod0_b)
```
在modMaxsum数组的第0/1/2个位置即为每个block对应数组mod 3 = 0/1/2的最大子序列和
最后在CPU上执行一次循环次数为numBlocks的最大子序列和合并，得到全局的最大子序列和

## Quantization
[pytorch Quantization Introduction](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)

[pytorch Quantization Tutorial](https://pytorch.org/docs/2.0/quantization.html)

[各种量化工具的差异](http://www.360doc.com/content/22/0127/12/7673502_1015090625.shtml)

## int8量化相关论文：

[Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf)

[Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation](https://arxiv.org/abs/2004.09602)

## 算子融合相关论文：

[Learning to Fuse](http://mlforsystems.org/assets/papers/neurips2019/learning_abdolrashidi_2019.pdf)

## AI系统资料：

https://github.com/chenzomi12/DeepLearningSystem
