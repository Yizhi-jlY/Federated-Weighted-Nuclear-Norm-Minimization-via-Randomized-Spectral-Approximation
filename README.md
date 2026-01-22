# FedWNNM: A Federated Framework for Matrix Completion with Provable Privacy-Accuracy Trade-offs

[![MATLAB](https://img.shields.io/badge/MATLAB-R2023a+-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

本代码库为论文 **"FedWNNM: A Federated Framework for Matrix Completion with Provable Privacy-Accuracy Trade-offs"** 的补充材料，包含重现论文中所有实验结果所需的源代码和相关资源。

## 📋 目录

- [系统要求](#系统要求)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [实验说明](#实验说明)
  - [实验1：相位转换分析](#实验1相位转换分析)
  - [实验2：图像修复应用](#实验2图像修复应用)
  - [实验3：隐私保护SVD子程序分析](#实验3隐私保护svd子程序分析)
  - [实验4：合成数据数值验证](#实验4合成数据数值验证)
  - [附录1：鲁棒性和超参数敏感性](#附录1鲁棒性和超参数敏感性)
- [核心算法](#核心算法)
- [引用](#引用)
- [许可证](#许可证)

## 💻 系统要求

- **软件**: MATLAB R2023a 或更高版本
- **操作系统**: 代码已在 Windows 11 上测试，预期可在 macOS 和 Linux 上运行
- **依赖项**: 代码是独立的。奇异值分解所需的 `PROPACK` 库已包含在 `code/Fed_WNNM/PROPACK` 和 `code/WNNM/PROPACK` 目录中
- **硬件建议**: 
  - CPU: 多核处理器（实验使用并行计算）
  - 内存: 8GB+ RAM
  - 存储: 至少 2GB 可用空间用于数据集和结果

## 📁 项目结构

```
.
├── Appendix1：Robustness and Hyperparameter Sensitivity of FedWNNM/
│   └── main.m                          # 鲁棒性和敏感性分析脚本（附录E中的图表）
│
├── Experement1：Phase Transition Analysis/
│   ├── main.m                          # 相位转换实验主脚本（图2）
│   ├── analyze_and_plot_results.m      # 结果分析和绘图函数
│   ├── load_all_results.m              # 结果加载函数
│   └── load_full_results.m             # 完整结果加载函数
│
├── Experement2：Application to Image Inpainting/
│   ├── main_dataset.m                  # 图像修复主脚本（表1，图3）
│   ├── main_color.m                    # 彩色图像处理脚本
│   ├── main_show_curve.m               # 收敛曲线展示
│   ├── process_images.m                # 图像批处理函数
│   └── select_zoom_region.m            # 区域选择工具
│
├── Experement3：Analysis of the Privacy-Preserving SVD Subroutine/
│   ├── main.m                          # PPF-SVD分析主脚本（表2，图4）
│   ├── experiment_dp_tradeoff.m        # 差分隐私权衡实验
│   ├── experiment_rank_adaptation_attack.m  # 秩自适应攻击实验
│   ├── reconstruction_attack_demo.m    # 重构攻击演示
│   ├── experiment_rho_effect.m         # ρ参数效果分析
│   ├── experiment_p_over_effect.m      # 过采样参数效果分析
│   ├── analyze_privacy_protection.m    # 隐私保护分析
│   ├── compute_errors.m                # 误差计算函数
│   ├── federated_randomized_svd_parallel.m  # 并行联邦随机SVD
│   └── federated_randomized_svd.m      # 联邦随机SVD实现
│
├── Experement4：Numerical Validation on Synthetic Data/
│   └── main.m                          # 合成数据实验主脚本（表3）
│
├── code/                               # 核心算法源代码
│   ├── AltGD/                          # 基线矩阵分解方法
│   │   ├── AltGD.m                     # 交替梯度下降算法
│   │   ├── altGDMin_T.m                # 带截断的交替梯度下降
│   │   ├── altGDMinCntrl_T.m           # 中心化版本
│   │   ├── altMinCntrl_T.m             # 中心化交替最小化
│   │   ├── altMinPrvt_T.m              # 隐私保护交替最小化
│   │   ├── communication_volume.m      # 通信量计算
│   │   ├── fedSvd_UV.m                 # 联邦SVD（UV分解）
│   │   └── fedSvd.m                    # 联邦SVD主函数
│   │
│   ├── Fed_WNNM/                       # 提出的FedWNNM方法
│   │   ├── FedWNNM_MC.m                # FedWNNM矩阵补全主函数
│   │   ├── ClosedWNNM.m                # 闭式WNNM求解器
│   │   ├── FR_svd_parallel.m           # 并行PPF-SVD实现
│   │   └── PROPACK/                    # PROPACK SVD库
│   │
│   ├── WNNM/                           # 中心化WNNM基线方法
│   │   ├── WNNM_MC.m                   # WNNM矩阵补全
│   │   ├── ClosedWNNM.m                # 闭式WNNM求解器
│   │   ├── relative_error.m            # 相对误差计算
│   │   └── PROPACK/                    # PROPACK SVD库
│   │
│   └── utils/                          # 工具函数
│       ├── display_results_table.m     # 结果表格显示
│       ├── plot_convergence_curves.m   # 收敛曲线绘制
│       └── Evaluation/                 # 评估指标函数
│
├── datasets/                           # 数据集
│   └── cbsd68t/                        # CBSD68图像数据集（测试集）
│       ├── 0000.png
│       ├── 0010.png
│       ├── 0013.png
│       ├── 0018.png
│       ├── 0027.png
│       └── 0046.png
│
├── CBSD68.txt                          # CBSD68数据集信息
├── main.m                              # 主角度示意图生成脚本
├── main2.m                             # 主角度示意图生成脚本（变体）
├── Readme.txt                          # 原始说明文件
└── README.md                           # 本文件
```

## 🚀 快速开始

### 1. 克隆或下载代码库

```bash
git clone https://github.com/yourusername/FedWNNM.git
cd FedWNNM
```

### 2. 启动MATLAB

在MATLAB中打开项目根目录。

### 3. 运行实验

每个实验都有独立的 `main.m` 或 `main_dataset.m` 脚本。导航到相应的实验文件夹并运行主脚本即可。

```matlab
% 示例：运行实验1
cd 'Experement1：Phase Transition Analysis'
main
```

## 🔬 实验说明

### 实验1：相位转换分析

**对应论文**: 图2

**实验目的**: 评估不同算法在不同矩阵秩和缺失率下的恢复性能。

**运行方法**:
```matlab
cd 'Experement1：Phase Transition Analysis'
main
```

**输出**:
- 所有对比算法的相位转换图
- 结果保存在 `results/Experiment1/<timestamp>/` 目录
- 控制台显示性能统计信息

**参数说明**:
- `params.m`: 矩阵行数（默认: 100）
- `params.n`: 矩阵列数（默认: 100）
- `params.mc`: 蒙特卡洛模拟次数（默认: 20）
- `missing_rates`: 缺失率范围（默认: 0.1 到 0.9）
- `ranks`: 矩阵秩范围（默认: 1 到 9）

### 实验2：图像修复应用

**对应论文**: 表1 & 图3

**实验目的**: 将FedWNNM和基线方法应用于CBSD68数据集的图像修复任务。

**运行方法**:
```matlab
cd 'Experement2：Application to Image Inpainting'
main_dataset
```

**输出**:
- 修复后的图像保存在 `Experiment4_ImageInpainting_Batch_Color/` 目录
- 控制台打印PSNR和SSIM值汇总表（对应表1）
- 生成并保存收敛曲线（对应图3的子图）

**支持的算法**:
- FedWNNM（提出的方法）
- WNNM（中心化基线）
- AltGD系列（AltGD, AltGDMin, AltMinCntrl, AltMinPrvt）

**数据集**: CBSD68测试集的6张代表性图像（256×256像素）

### 实验3：隐私保护SVD子程序分析

**对应论文**: 表2 & 图4

**实验目的**: 分析PPF-SVD（Privacy-Preserving Federated SVD）子程序的隐私-效用权衡。

**运行方法**:
```matlab
cd 'Experement3：Analysis of the Privacy-Preserving SVD Subroutine'
main
```

**输出**:
- 重构误差指标和奇异值比较图（对应图4）
- 不同ρ值的定量结果（对应表2）
- 隐私保护效果分析图

**包含的子实验**:
1. **ρ参数效果**: 分析对角衰减因子ρ对精度的影响
2. **过采样参数效果**: 评估过采样参数p_over的作用
3. **隐私-效用权衡**: 差分隐私参数ε的扫描分析
4. **秩自适应攻击**: 比较FedWNNM vs. AltGDMin在秩误指定下的鲁棒性
5. **视觉重构攻击**: 展示原始vs.无DP vs.有DP的图像重构

**差分隐私配置**:
- 使用裁剪+高斯噪声实现(ε,δ)-差分隐私
- 通过 `dp_*` 参数配置（在 `federated_randomized_svd_parallel.m` 和 `FR_svd_parallel.m` 中）

### 实验4：合成数据数值验证

**对应论文**: 表3

**实验目的**: 在合成低秩矩阵上对所有算法进行基准测试。

**运行方法**:
```matlab
cd 'Experement4：Numerical Validation on Synthetic Data'
main
```

**输出**:
- 运行30次蒙特卡洛试验
- 打印平均相对误差和计算时间汇总表（对应表3）
- 结果保存在 `results/Experiment4/<timestamp>/` 目录

**评估指标**:
- 相对恢复误差: `||L_hat - L_true||_F / ||L_true||_F`
- 计算时间（秒）
- 收敛性统计

### 附录1：鲁棒性和超参数敏感性

**对应论文**: 附录E中的图表

**实验目的**: 评估FedWNNM对关键超参数的敏感性。

**运行方法**:
```matlab
cd 'Appendix1：Robustness and Hyperparameter Sensitivity of FedWNNM'
main
```

**分析的超参数**:
- `C`: WNNM权重参数
- `p_over`: 过采样参数
- `rho`: 对角衰减因子
- `q`: 幂迭代次数
- `p`: 客户端数量

## 🧮 核心算法

### FedWNNM算法

**主函数**: [`code/Fed_WNNM/FedWNNM_MC.m`](code/Fed_WNNM/FedWNNM_MC.m)

**算法描述**: FedWNNM（Federated Weighted Nuclear Norm Minimization）是一种联邦学习框架，用于在分布式环境中进行矩阵补全，同时保护数据隐私。

**关键特性**:
- ✅ 分布式计算：数据分布在多个客户端，无需集中存储
- ✅ 隐私保护：通过PPF-SVD子程序保护客户端数据隐私
- ✅ 高效通信：使用随机化SVD减少通信开销
- ✅ 可证明的隐私-精度权衡

**输入参数**:
```matlab
result = FedWNNM_MC(data, mask, parameters)
```
- `data`: m×n 观测矩阵（带缺失值）
- `mask`: m×n 二值掩码（1表示观测，0表示缺失）
- `parameters`: 参数结构体
  - `p`: 联邦客户端数量（默认: 4）
  - `C`: WNNM权重参数（默认: 1）
  - `tol`: 收敛容差（默认: 1e-7）
  - `maxiter`: 最大迭代次数（默认: 500）
  - `p_over`: 过采样参数（默认: 10）
  - `rho`: 对角衰减因子（默认: 1）
  - `q`: 幂迭代次数（默认: 20）

**输出结果**:
- `A_hat`: 恢复的低秩矩阵
- `E_hat`: 恢复的稀疏分量
- `iteration_count`: 迭代次数
- `total_time`: 总执行时间
- `relative_error`: 相对恢复误差
- `communication_volumes`: 每轮通信量（MB）

### PPF-SVD子程序

**主函数**: [`code/Fed_WNNM/FR_svd_parallel.m`](code/Fed_WNNM/FR_svd_parallel.m)

**算法描述**: Privacy-Preserving Federated Randomized SVD是一种联邦学习环境下的随机化奇异值分解方法，通过添加对角衰减和可选的差分隐私噪声来保护数据隐私。

**关键特性**:
- 🔐 隐私保护：通过对角衰减矩阵和差分隐私机制
- ⚡ 高效计算：使用随机化方法降低计算复杂度
- 🔄 并行化：支持多客户端并行计算
- 📊 可配置隐私级别：通过ρ和ε参数调节隐私-效用权衡

### 基线算法

1. **WNNM** (Weighted Nuclear Norm Minimization)
   - 文件: [`code/WNNM/WNNM_MC.m`](code/WNNM/WNNM_MC.m)
   - 描述: 中心化的加权核范数最小化方法

2. **AltGD** (Alternating Gradient Descent)
   - 文件: [`code/AltGD/AltGD.m`](code/AltGD/AltGD.m)
   - 描述: 交替梯度下降矩阵分解方法

3. **AltGDMin** (Alternating Gradient Descent with Truncation)
   - 文件: [`code/AltGD/altGDMin_T.m`](code/AltGD/altGDMin_T.m)
   - 描述: 带截断的交替梯度下降

4. **AltMinCntrl** (Alternating Minimization Centralized)
   - 文件: [`code/AltGD/altMinCntrl_T.m`](code/AltGD/altMinCntrl_T.m)
   - 描述: 中心化交替最小化

5. **AltMinPrvt** (Alternating Minimization Private)
   - 文件: [`code/AltGD/altMinPrvt_T.m`](code/AltGD/altMinPrvt_T.m)
   - 描述: 隐私保护交替最小化

## 📊 评估指标

代码实现了以下评估指标：

- **相对误差** (Relative Error): `||L_hat - L_true||_F / ||L_true||_F`
- **PSNR** (Peak Signal-to-Noise Ratio): 图像质量评估
- **SSIM** (Structural Similarity Index): 结构相似性评估
- **子空间距离** (Subspace Distance): 主子空间恢复精度
- **奇异值误差** (Singular Value Error): 奇异值恢复精度
- **通信量** (Communication Volume): 联邦学习通信开销（MB）
- **计算时间** (Computation Time): 算法运行时间

## 🔧 常见问题

### 1. PROPACK编译问题

如果遇到PROPACK MEX文件编译问题，请：
- 确保安装了兼容的C/Fortran编译器
- 运行 `mex -setup` 配置编译器
- PROPACK库已包含预编译的MEX文件（Windows x86/x64）

### 2. 内存不足

对于大规模矩阵实验，建议：
- 减少 `params.mc`（蒙特卡洛试验次数）
- 减少矩阵维度 `params.m` 和 `params.n`
- 关闭并行计算池以节省内存

### 3. 运行时间过长

优化建议：
- 减少最大迭代次数 `params.maxiter`
- 增加收敛容差 `params.tol`
- 使用并行计算（MATLAB Parallel Computing Toolbox）
- 在 `run_and_analyze` 模式下先运行小规模测试

### 4. 路径管理

脚本使用动态路径管理策略：
- 自动添加必要的代码路径
- 运行后自动清理路径
- 如遇路径问题，检查 `codeFolderPath` 变量设置

## 📝 代码风格和注释

所有代码都包含详细的注释：
- **函数头部**: 完整的输入/输出参数说明
- **算法步骤**: 逐步解释算法实现
- **参数说明**: 所有可配置参数的含义和默认值
- **示例用法**: 关键函数包含使用示例

## 🤝 贡献

欢迎提交问题报告和改进建议！

## 📧 联系方式

如有任何问题，请通过以下方式联系：
- 📧 Email: [your.email@example.com]
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/FedWNNM/issues)

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- PROPACK库：用于高效的奇异值分解
- CBSD68数据集：用于图像修复实验
- MATLAB社区：提供的优秀工具和资源

## 📚 引用

如果您在研究中使用了本代码，请引用我们的论文：

```bibtex
@article{fedwnnm2024,
  title={FedWNNM: A Federated Framework for Matrix Completion with Provable Privacy-Accuracy Trade-offs},
  author={Your Name and Collaborators},
  journal={Journal Name},
  year={2024},
  volume={XX},
  pages={XXX-XXX}
}
```

---

**最后更新**: 2026年1月

**版本**: 1.0.0
