# FedWNNM: A Federated Framework for Matrix Completion with Provable Privacy-Accuracy Trade-offs

[![MATLAB](https://img.shields.io/badge/MATLAB-R2023a+-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[中文文档](README.md) | **English**

This repository provides supplementary code for the paper **"FedWNNM: A Federated Framework for Matrix Completion with Provable Privacy-Accuracy Trade-offs"**. It contains source code and resources needed to reproduce all experimental results presented in the paper.

## 📋 Table of Contents

- [System Requirements](#system-requirements)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Experiments](#experiments)
  - [Experiment 1: Phase Transition Analysis](#experiment-1-phase-transition-analysis)
  - [Experiment 2: Application to Image Inpainting](#experiment-2-application-to-image-inpainting)
  - [Experiment 3: Privacy-Preserving SVD Analysis](#experiment-3-privacy-preserving-svd-analysis)
  - [Experiment 4: Synthetic Data Validation](#experiment-4-synthetic-data-validation)
  - [Appendix 1: Robustness and Hyperparameter Sensitivity](#appendix-1-robustness-and-hyperparameter-sensitivity)
- [Core Algorithms](#core-algorithms)
- [Citation](#citation)
- [License](#license)

## 💻 System Requirements

- **Software**: MATLAB R2023a or higher
- **Operating System**: Tested on Windows 11, expected to work on macOS and Linux
- **Dependencies**: Self-contained. The `PROPACK` library for SVD is included in `code/Fed_WNNM/PROPACK` and `code/WNNM/PROPACK`
- **Hardware Recommendations**: 
  - CPU: Multi-core processor (experiments use parallel computing)
  - Memory: 8GB+ RAM
  - Storage: At least 2GB free space for datasets and results

## 📁 Project Structure

```
.
├── Appendix1：Robustness and Hyperparameter Sensitivity of FedWNNM/
│   └── main.m                          # Robustness and sensitivity analysis (Appendix E)
│
├── Experement1：Phase Transition Analysis/
│   ├── main.m                          # Phase transition experiment script (Figure 2)
│   ├── analyze_and_plot_results.m      # Result analysis and plotting
│   ├── load_all_results.m              # Result loading function
│   └── load_full_results.m             # Full result loading function
│
├── Experement2：Application to Image Inpainting/
│   ├── main_dataset.m                  # Image inpainting main script (Table 1, Figure 3)
│   ├── main_color.m                    # Color image processing
│   ├── main_show_curve.m               # Convergence curve display
│   ├── process_images.m                # Batch image processing
│   └── select_zoom_region.m            # Region selection tool
│
├── Experement3：Analysis of the Privacy-Preserving SVD Subroutine/
│   ├── main.m                          # PPF-SVD analysis main script (Table 2, Figure 4)
│   ├── experiment_dp_tradeoff.m        # Differential privacy trade-off experiment
│   ├── experiment_rank_adaptation_attack.m  # Rank adaptation attack experiment
│   ├── reconstruction_attack_demo.m    # Reconstruction attack demonstration
│   ├── experiment_rho_effect.m         # Rho parameter effect analysis
│   ├── experiment_p_over_effect.m      # Oversampling parameter analysis
│   ├── analyze_privacy_protection.m    # Privacy protection analysis
│   ├── compute_errors.m                # Error computation function
│   ├── federated_randomized_svd_parallel.m  # Parallel federated randomized SVD
│   └── federated_randomized_svd.m      # Federated randomized SVD
│
├── Experement4：Numerical Validation on Synthetic Data/
│   └── main.m                          # Synthetic data experiment script (Table 3)
│
├── code/                               # Core algorithm source code
│   ├── AltGD/                          # Baseline matrix factorization methods
│   │   ├── AltGD.m                     # Alternating gradient descent
│   │   ├── altGDMin_T.m                # AltGD with truncation
│   │   ├── altGDMinCntrl_T.m           # Centralized version
│   │   ├── altMinCntrl_T.m             # Centralized alternating minimization
│   │   ├── altMinPrvt_T.m              # Private alternating minimization
│   │   ├── communication_volume.m      # Communication volume computation
│   │   ├── fedSvd_UV.m                 # Federated SVD (UV decomposition)
│   │   └── fedSvd.m                    # Federated SVD main function
│   │
│   ├── Fed_WNNM/                       # Proposed FedWNNM method
│   │   ├── FedWNNM_MC.m                # FedWNNM matrix completion main function
│   │   ├── ClosedWNNM.m                # Closed-form WNNM solver
│   │   ├── FR_svd_parallel.m           # Parallel PPF-SVD implementation
│   │   └── PROPACK/                    # PROPACK SVD library
│   │
│   ├── WNNM/                           # Centralized WNNM baseline
│   │   ├── WNNM_MC.m                   # WNNM matrix completion
│   │   ├── ClosedWNNM.m                # Closed-form WNNM solver
│   │   ├── relative_error.m            # Relative error computation
│   │   └── PROPACK/                    # PROPACK SVD library
│   │
│   └── utils/                          # Utility functions
│       ├── display_results_table.m     # Result table display
│       ├── plot_convergence_curves.m   # Convergence curve plotting
│       └── Evaluation/                 # Evaluation metric functions
│
├── datasets/                           # Datasets
│   └── cbsd68t/                        # CBSD68 image dataset (test set)
│       ├── 0000.png
│       ├── 0010.png
│       ├── 0013.png
│       ├── 0018.png
│       ├── 0027.png
│       └── 0046.png
│
├── CBSD68.txt                          # CBSD68 dataset information
├── main.m                              # Principal angles diagram generation
├── main2.m                             # Principal angles diagram (variant)
├── Readme.txt                          # Original readme file
├── README.md                           # This file (Chinese)
└── README_EN.md                        # This file (English)
```

## 🚀 Quick Start

### 1. Clone or Download Repository

```bash
git clone https://github.com/yourusername/FedWNNM.git
cd FedWNNM
```

### 2. Launch MATLAB

Open the project root directory in MATLAB.

### 3. Run Experiments

Each experiment has an independent `main.m` or `main_dataset.m` script. Navigate to the corresponding experiment folder and run the main script.

```matlab
% Example: Run Experiment 1
cd 'Experement1：Phase Transition Analysis'
main
```

## 🔬 Experiments

### Experiment 1: Phase Transition Analysis

**Paper Reference**: Figure 2

**Purpose**: Evaluate recovery performance of different algorithms under varying matrix ranks and missing rates.

**How to Run**:
```matlab
cd 'Experement1：Phase Transition Analysis'
main
```

**Outputs**:
- Phase transition plots for all compared algorithms
- Results saved in `results/Experiment1/<timestamp>/` directory
- Performance statistics displayed in console

**Parameters**:
- `params.m`: Number of matrix rows (default: 100)
- `params.n`: Number of matrix columns (default: 100)
- `params.mc`: Number of Monte Carlo simulations (default: 20)
- `missing_rates`: Missing rate range (default: 0.1 to 0.9)
- `ranks`: Matrix rank range (default: 1 to 9)

### Experiment 2: Application to Image Inpainting

**Paper Reference**: Table 1 & Figure 3

**Purpose**: Apply FedWNNM and baseline methods to image inpainting tasks on the CBSD68 dataset.

**How to Run**:
```matlab
cd 'Experement2：Application to Image Inpainting'
main_dataset
```

**Outputs**:
- Recovered images saved in `Experiment4_ImageInpainting_Batch_Color/` directory
- PSNR and SSIM summary table printed to console (corresponding to Table 1)
- Convergence curves generated and saved (corresponding to Figure 3 subplots)

**Supported Algorithms**:
- FedWNNM (proposed method)
- WNNM (centralized baseline)
- AltGD series (AltGD, AltGDMin, AltMinCntrl, AltMinPrvt)

**Dataset**: 6 representative images from CBSD68 test set (256×256 pixels)

### Experiment 3: Privacy-Preserving SVD Analysis

**Paper Reference**: Table 2 & Figure 4

**Purpose**: Analyze the privacy-utility trade-off of the PPF-SVD (Privacy-Preserving Federated SVD) subroutine.

**How to Run**:
```matlab
cd 'Experement3：Analysis of the Privacy-Preserving SVD Subroutine'
main
```

**Outputs**:
- Reconstruction error metrics and singular value comparison plots (corresponding to Figure 4)
- Quantitative results for different ρ values (corresponding to Table 2)
- Privacy protection effectiveness analysis plots

**Sub-experiments Included**:
1. **Rho Parameter Effect**: Analyze the impact of diagonal decay factor ρ on accuracy
2. **Oversampling Parameter Effect**: Evaluate the role of oversampling parameter p_over
3. **Privacy-Utility Trade-off**: Differential privacy parameter ε sweep analysis
4. **Rank Adaptation Attack**: Compare robustness of FedWNNM vs. AltGDMin under rank misspecification
5. **Visual Reconstruction Attack**: Demonstrate original vs. no-DP vs. DP image reconstructions

**Differential Privacy Configuration**:
- Uses clipping + Gaussian noise for (ε,δ)-differential privacy
- Configured via `dp_*` parameters (in `federated_randomized_svd_parallel.m` and `FR_svd_parallel.m`)

### Experiment 4: Synthetic Data Validation

**Paper Reference**: Table 3

**Purpose**: Benchmark performance of all algorithms on synthetic low-rank matrices.

**How to Run**:
```matlab
cd 'Experement4：Numerical Validation on Synthetic Data'
main
```

**Outputs**:
- Runs 30 Monte Carlo trials
- Prints summary table of average relative error and computation time (corresponding to Table 3)
- Results saved in `results/Experiment4/<timestamp>/` directory

**Evaluation Metrics**:
- Relative recovery error: `||L_hat - L_true||_F / ||L_true||_F`
- Computation time (seconds)
- Convergence statistics

### Appendix 1: Robustness and Hyperparameter Sensitivity

**Paper Reference**: Figures in Appendix E

**Purpose**: Evaluate the sensitivity of FedWNNM to key hyperparameters.

**How to Run**:
```matlab
cd 'Appendix1：Robustness and Hyperparameter Sensitivity of FedWNNM'
main
```

**Analyzed Hyperparameters**:
- `C`: WNNM weight parameter
- `p_over`: Oversampling parameter
- `rho`: Diagonal decay factor
- `q`: Number of power iterations
- `p`: Number of clients

## 🧮 Core Algorithms

### FedWNNM Algorithm

**Main Function**: [`code/Fed_WNNM/FedWNNM_MC.m`](code/Fed_WNNM/FedWNNM_MC.m)

**Description**: FedWNNM (Federated Weighted Nuclear Norm Minimization) is a federated learning framework for matrix completion in distributed environments while protecting data privacy.

**Key Features**:
- ✅ Distributed Computing: Data distributed across multiple clients without centralized storage
- ✅ Privacy Protection: Protects client data privacy through PPF-SVD subroutine
- ✅ Efficient Communication: Uses randomized SVD to reduce communication overhead
- ✅ Provable Privacy-Accuracy Trade-offs

**Input Parameters**:
```matlab
result = FedWNNM_MC(data, mask, parameters)
```
- `data`: m×n observed matrix (with missing values)
- `mask`: m×n binary mask (1 for observed, 0 for missing)
- `parameters`: Parameter structure
  - `p`: Number of federated clients (default: 4)
  - `C`: WNNM weight parameter (default: 1)
  - `tol`: Convergence tolerance (default: 1e-7)
  - `maxiter`: Maximum iterations (default: 500)
  - `p_over`: Oversampling parameter (default: 10)
  - `rho`: Diagonal decay factor (default: 1)
  - `q`: Number of power iterations (default: 20)

**Output Results**:
- `A_hat`: Recovered low-rank matrix
- `E_hat`: Recovered sparse component
- `iteration_count`: Number of iterations
- `total_time`: Total execution time
- `relative_error`: Relative recovery error
- `communication_volumes`: Communication volume per round (MB)

### PPF-SVD Subroutine

**Main Function**: [`code/Fed_WNNM/FR_svd_parallel.m`](code/Fed_WNNM/FR_svd_parallel.m)

**Description**: Privacy-Preserving Federated Randomized SVD is a randomized singular value decomposition method in federated learning environments, protecting data privacy through diagonal decay and optional differential privacy noise.

**Key Features**:
- 🔐 Privacy Protection: Through diagonal decay matrix and differential privacy mechanisms
- ⚡ Efficient Computation: Uses randomization to reduce computational complexity
- 🔄 Parallelization: Supports multi-client parallel computation
- 📊 Configurable Privacy Level: Adjustable privacy-utility trade-off via ρ and ε parameters

### Baseline Algorithms

1. **WNNM** (Weighted Nuclear Norm Minimization)
   - File: [`code/WNNM/WNNM_MC.m`](code/WNNM/WNNM_MC.m)
   - Description: Centralized weighted nuclear norm minimization

2. **AltGD** (Alternating Gradient Descent)
   - File: [`code/AltGD/AltGD.m`](code/AltGD/AltGD.m)
   - Description: Alternating gradient descent matrix factorization

3. **AltGDMin** (Alternating Gradient Descent with Truncation)
   - File: [`code/AltGD/altGDMin_T.m`](code/AltGD/altGDMin_T.m)
   - Description: Alternating gradient descent with truncation

4. **AltMinCntrl** (Alternating Minimization Centralized)
   - File: [`code/AltGD/altMinCntrl_T.m`](code/AltGD/altMinCntrl_T.m)
   - Description: Centralized alternating minimization

5. **AltMinPrvt** (Alternating Minimization Private)
   - File: [`code/AltGD/altMinPrvt_T.m`](code/AltGD/altMinPrvt_T.m)
   - Description: Privacy-preserving alternating minimization

## 📊 Evaluation Metrics

The code implements the following evaluation metrics:

- **Relative Error**: `||L_hat - L_true||_F / ||L_true||_F`
- **PSNR** (Peak Signal-to-Noise Ratio): Image quality assessment
- **SSIM** (Structural Similarity Index): Structural similarity assessment
- **Subspace Distance**: Principal subspace recovery accuracy
- **Singular Value Error**: Singular value recovery accuracy
- **Communication Volume**: Federated learning communication overhead (MB)
- **Computation Time**: Algorithm execution time

## 🔧 Troubleshooting

### 1. PROPACK Compilation Issues

If you encounter PROPACK MEX file compilation issues:
- Ensure a compatible C/Fortran compiler is installed
- Run `mex -setup` to configure the compiler
- PROPACK library includes pre-compiled MEX files (Windows x86/x64)

### 2. Out of Memory

For large-scale matrix experiments, consider:
- Reducing `params.mc` (number of Monte Carlo trials)
- Reducing matrix dimensions `params.m` and `params.n`
- Closing parallel computing pool to save memory

### 3. Long Execution Time

Optimization suggestions:
- Reduce maximum iterations `params.maxiter`
- Increase convergence tolerance `params.tol`
- Use parallel computing (MATLAB Parallel Computing Toolbox)
- Run small-scale tests first in `run_and_analyze` mode

### 4. Path Management

Scripts use dynamic path management strategy:
- Automatically adds necessary code paths
- Automatically cleans up paths after execution
- If path issues occur, check `codeFolderPath` variable settings

## 📝 Code Style and Comments

All code includes detailed comments:
- **Function Headers**: Complete input/output parameter descriptions
- **Algorithm Steps**: Step-by-step algorithm implementation explanations
- **Parameter Descriptions**: Meanings and default values of all configurable parameters
- **Usage Examples**: Key functions include usage examples

## 🤝 Contributing

Bug reports and improvement suggestions are welcome!

## 📧 Contact

For any questions, please contact:
- 📧 Email: [your.email@example.com]
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/FedWNNM/issues)

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PROPACK library: For efficient singular value decomposition
- CBSD68 dataset: For image inpainting experiments
- MATLAB community: For excellent tools and resources

## 📚 Citation

If you use this code in your research, please cite our paper:

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

**Last Updated**: January 2026

**Version**: 1.0.0
