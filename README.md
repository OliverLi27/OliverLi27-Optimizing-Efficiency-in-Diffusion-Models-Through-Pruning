# Optimizing Efficiency in Diffusion Models through Pruning

## Introduction
> **Optimizing Efficiency in Diffusion Models through Pruning**
> * Building a stochastically pruned diffusion model using dependency groups
> * Training 100k step on the CIFAR-10 

## Pruning with DDIM codebase

This example shows how to prune a DDPM model pre-trained on CIFAR-10 using the [DDIM codebase](https://github.com/ermongroup/ddim). Since that [Huggingface Diffusers](https://github.com/huggingface/diffusers) do not support [``skip_type='quad'``](https://github.com/ermongroup/ddim/issues/3) in DDIM, you may get slightly worse FID scores with Diffusers for both pre-trained models (FID=4.5) and pruned models (FID=5.6). We are working on this to implement the quad strategy for Diffusers. 

```bash
cd exp_code
# Prune & Finetune
bash scripts/simple_cifar_Ran.sh 0.05 # the pre-trained model and data will be automatically prepared
# Fine-tuning
bash scripts/sample_cifar_ddpm_pruning.sh run/finetune_simple_v2/cifar10_ours_T=0.05.pth/logs/post_training/ckpt_100000.pth run/sample
```

For FID, please refer to [this section](https://github.com/VainF/Diff-Pruning#4-fid-score).  

Output:
```
Found 49984 files.
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:49<00:00,  7.97it/s]
FID:  5.712596900921312
```

## Some necessary operation

### 0. Requirements, Data and Pretrained Model

* Requirements
```bash
pip install -r requirements.txt
```

### 1. Sampling
**Pruned:** Sample and save images to *run/sample/ddpm_cifar10_pruned*
```bash
bash exp_code/scripts/sample_ddpm_cifar10_pruned.sh
```

**Pretrained:** Sample and save images to *run/sample/ddpm_cifar10_pretrained*
```bash
bash exp_code/cripts/sample_ddpm_cifar10_pretrained.sh
```

### 2. FID Score
This script was modified from https://github.com/mseitzer/pytorch-fid. 

```bash
# pre-compute the stats of CIFAR-10 dataset
python fid_score.py --save-stats data/cifar10_images run/fid_stats_cifar10.npz --device cuda:0 --batch-size 256
```

```bash
# Compute the FID score of sampled images
python fid_score.py run/sample/ddpm_cifar10_pruned run/fid_stats_cifar10.npz --device cuda:0 --batch-size 256
```