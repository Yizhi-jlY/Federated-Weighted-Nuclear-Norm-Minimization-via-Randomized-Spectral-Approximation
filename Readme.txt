# Supplementary Code for "FedWNNM: A Federated Framework for Matrix Completion with Provable Privacy-Accuracy Trade-offs"

This supplementary material provides the source code and necessary resources to reproduce the experimental results presented in the paper. The code is implemented in MATLAB.

## System Requirements

- **Software**: MATLAB R2023a or a compatible version.
- **Operating System**: The code has been tested on Windows 11 but is expected to be compatible with macOS and Linux.
- **Dependencies**: The code is self-contained. The `PROPACK` library, used for singular value decomposition, is included in the `code/Fed_WNNM/PROPACK` and `code/WNNM/PROPACK` directories.

## Directory Structure

The repository is organized as follows:

```
.
├── Appendix1：Robustness and Hyperparameter Sensitivity of FedWNNM/
│   └── main.m                # Script for robustness and sensitivity analysis (Figures in Appendix E)
├── Experement1：Phase Transition Analysis/
│   └── main.m                # Script for phase transition experiments (Figure 2)
├── Experement2：Application to Image Inpainting/
│   └── main_dataset.m        # Script for image inpainting experiments (Table 1, Figure 3)
├── Experement3：Analysis of the Privacy-Preserving SVD Subroutine/
│   └── main.m                # Script for PPF-SVD analysis (Table 2, Figure 4)
├── Experement4：Numerical Validation on Synthetic Data/
│   └── main.m                # Script for synthetic data experiments (Table 3)
├── code/                     # Source code for all algorithms and utility functions
│   ├── AltGD/                # Baseline matrix factorization methods
│   ├── Fed_WNNM/             # Proposed FedWNNM method and PPF-SVD
│   ├── WNNM/                 # Centralized WNNM method
│   └── utils/                # Utility functions for plotting and evaluation
├── datasets/                 # Image dataset used in the inpainting experiment
│   └── cbsd68t/
└── Supplementary Code for FedWNNM A Federated Framework for Matrix Completion with Provable Privacy-Accuracy Trade-offs.md  # This instruction file
```

## Running the Experiments

All experiments can be reproduced by running the `main.m` or `main_dataset.m` script within each corresponding experiment folder. The scripts will automatically save the generated figures and result files to a `results` subfolder within each experiment directory.

### 1. Experiment 1: Phase Transition Analysis (Figure 2)

This experiment evaluates the recovery performance of different algorithms under varying matrix ranks and missing rates.

- **To run**: Navigate to the `Experement1：Phase Transition Analysis/` directory and execute the `main.m` script.
- **Output**: The script will generate and save the phase transition plots for all compared algorithms, corresponding to Figure 2 in the paper.

### 2. Experiment 2: Application to Image Inpainting (Table 1 & Figure 3)

This experiment applies FedWNNM and baseline methods to an image inpainting task on the CBSD68 dataset.

- **To run**: Navigate to the `Experement2：Application to Image Inpainting/` directory and execute the `main_dataset.m` script.
- **Output**: This script will:
    - Run the inpainting task for all algorithms across various missing rates.
    - Save the recovered images in the `Experiment4_ImageInpainting_Batch_Color/` directory.
    - Print a summary table of PSNR and SSIM values to the console, corresponding to Table 1.
    - Generate and save the convergence curves, corresponding to the subplots in Figure 3.

### 3. Experiment 3: Analysis of the Privacy-Preserving SVD Subroutine (Table 2 & Figure 4)

This experiment analyzes the privacy-utility trade-off of the PPF-SVD subroutine by varying the privacy parameter `ρ`.

- **To run**: Navigate to the `Experement3：Analysis of the Privacy-Preserving SVD Subroutine/` directory and execute the `main.m` script.
- **Output**: The script will:
    - Generate and save plots for the reconstruction error metrics and singular value comparisons, corresponding to Figure 4.
    - Display the quantitative results for different `ρ` values in the console, corresponding to Table 2.

### 4. Experiment 4: Numerical Validation on Synthetic Data (Table 3)

This experiment benchmarks the performance of all algorithms on synthetic low-rank matrices.

- **To run**: Navigate to the `Experement4：Numerical Validation on Synthetic Data/` directory and execute the `main.m` script.
- **Output**: The script will run 30 Monte Carlo trials and then print a summary table of the average relative error and computation time to the console, corresponding to Table 3.

### 5. Appendix 1: Robustness and Hyperparameter Sensitivity (Figures in Appendix E)

This experiment evaluates the sensitivity of FedWNNM to its key hyperparameters.


### 6. New DP Experiments (Privacy Layer + Attacks)

We added a differential-privacy layer for the federated randomized SVD and three supporting experiments in `Experement3：Analysis of the Privacy-Preserving SVD Subroutine/`:

- **Privacy-Utility Trade-off**: In `main.m`, run the new epsilon sweep (uses `experiment_dp_tradeoff.m`); outputs `privacy_utility_tradeoff.png`/`.eps`.
- **Rank Adaptation Attack**: `experiment_rank_adaptation_attack.m` (also invoked from `main.m`) compares FedWNNM vs. AltGDMin with a mis-specified rank; outputs `rank_adaptation_attack.png`/`.eps`.
- **Visual Reconstruction Attack**: `reconstruction_attack_demo.m` shows original vs. no-DP vs. DP reconstructions on CBSD68; outputs `reconstruction_attack.png`/`.pdf`.

All DP logic uses clipping + Gaussian noise for $(\epsilon, \delta)$-DP; configure via the new `dp_*` parameters in `federated_randomized_svd_parallel` and `FR_svd_parallel`.

## Code Description

- **/code**: This directory contains the core implementation of our proposed method and all baselines.
  - `Fed_WNNM/FedWNNM_MC.m` is the main function for our proposed federated algorithm.
  - `Fed_WNNM/FR_svd_parallel.m` is the implementation of the PPF-SVD subroutine.
  - `AltGD/` and `WNNM/` contain the implementations for the baseline methods.
  - `utils/` contains helper functions for evaluation metrics (e.g., `relative_error.m`, `psnr_image.m`) and plotting.
- **/datasets**: This directory contains the images from the CBSD68 dataset used for the image inpainting experiment.