# DMNSP: Dynamic Multi-Layer Null Space Projection for Vision-Language Continual Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of the paper "Dynamic Multi-Layer Null Space Projection for Vision-Language Continual Learning" (ICCV 2025) in PyTorch.

## 🎯 Abstract

Vision-Language Models (VLM) have emerged as a highly promising approach for Continual Learning (CL) due to their powerful generalized features. While adapter-based VLM can exploit both task-specific and task-agnostic features, current CL methods have largely overlooked the distinct and evolving parameter distributions in visual and language modalities, which are found crucial for effectively mitigating catastrophic forgetting.In this study, we find that the **visual modality experiences a broader parameter distribution and greater variance** during class increments than the textual modality, leading to higher vulnerability to forgetting. Consequently, we handle the branches of the two modalities asymmetrically.

### Key Contributions

- 🔍 **Asymmetric Modality Handling**: We propose handling visual and language modalities differently based on their distinct parameter distribution characteristics
- 🚀 **Multi-layer Null Space Projection**: A novel strategy applied only to the visual modality branch to restrict parameter updates within specific subspaces
- ⚖️ **Dynamic Projection Coefficient**: Precise control of gradient projection magnitude for optimal stability-plasticity balance

## 🛠️ Installation

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/RL-VIG/DMNSP.git
cd DMNSP

# Install dependencies
pip install -r requirements.txt
```

## 📊 Datasets

The framework supports the following datasets for class incremental learning:

- **CIFAR100**: 100 classes, various incremental settings (2-2, 5-5, 10-10)
- **TinyImageNet**: 200 classes, incremental settings (200-100-5, 200-100-10, 200-100-20)
- To be updated......

### Data Preparation

1. The datasets will be automatically downloaded when running experiments
2. Update the `dataset_root` path in your configuration files or command line
3. Ensure sufficient disk space for dataset storage

## 🚀 Quick Start

### Basic Usage

```bash
# Run CIFAR100 with 10 initial classes and 10 incremental classes
sh run_cifar100-10-10.sh

# Or run with custom parameters
python main.py \
    --config-path ./configs/class \
    --config-name cifar100_10-10.yaml \
    dataset_root="/path/to/your/data" \
    class_order="./class_orders/cifar100.yaml"
```

### Configuration Options

The project uses Hydra for configuration management. Key parameters include:

```yaml
# Model settings
model_name: "ViT-B/16"              # CLIP model variant
prompt_template: "a bad photo of a {}."  # Text prompt template

# Training settings
batch_size: 128                     # Training batch size
lr: 1e-3                           # Learning rate
weight_decay: 0.0                   # Weight decay
ls: 0.0                            # Label smoothing

# Incremental learning settings
initial_increment: 10               # Initial number of classes
increment: 10                       # Classes per incremental step
method: "DMNSP"                     # Method name
```

## 🔧 Advanced Usage

### Custom Datasets

To add support for new datasets:

1. Add dataset configuration in `continual_clip/datasets.py`
2. Create corresponding class order file in `class_orders/`
3. Add configuration YAML in `configs/class/`

### Hyperparameter Tuning

Key hyperparameters for optimization:

- `TOP_SELECT`: Number of top singular vectors to select (default: 1)
- `TOP_K_RATIO`: Ratio for top-k similarity selection (default: 0.1)
- `LAMBDA_SCALE`: Scaling factor for dynamic coefficients (default: 30)

**Note**: For selecting the basis vectors of the null space, fixed values can be used.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{Kang2025DMNSP,
  title={Dynamic Multi-Layer Null Space Projection for Vision-Language Continual Learning},
  author={Borui Kang, Lei Wang, Zhiping Wu, Tao Feng, Yawen Li, Yang Gao, Wenbin Li},
  journal={ICCV},
  year={2025}
}
```

## 📞 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact the authors at [kangborui.cn@gmail.com]

---

**Note**: This implementation is for research purposes. Please ensure you comply with the respective licenses of the datasets and models used.



