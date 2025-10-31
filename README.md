# Atlas E-Commerce Clothing Classifier

Complete deep learning project for automated clothing categorization using the Atlas dataset.

## Project Overview

This project implements a CNN-based image classifier for e-commerce clothing categorization using:
- **Dataset**: Atlas E-commerce Clothing Dataset (78,358 images, 52 categories)
- **Model**: ResNet34 with custom classification head
- **Framework**: PyTorch
- **Deployment**: Flask web application

## Dataset

**Kaggle Link**: https://www.kaggle.com/datasets/silverstone1903/atlas-e-commerce-clothing-product-categorization

- Total Images: 78,358
- Categories: 52 leaf nodes in 3-level hierarchy
- Taxonomy: Gender > Clothing Type > Specific Category
- Expected Accuracy: ~92%

## Project Structure

```
atlas_clothing_classifier/
├── data/
│   ├── raw/                    # Downloaded dataset
│   ├── processed/              # Preprocessed images
│   └── splits/                 # Train/val/test CSV files
├── models/
│   ├── checkpoints/            # Training checkpoints
│   └── final/                  # Final trained models
├── src/
│   ├── data/                   # Data loading and preprocessing
│   ├── models/                 # Model architectures
│   ├── training/               # Training scripts
│   ├── evaluation/             # Evaluation and metrics
│   └── inference/              # Inference pipeline
├── notebooks/                  # Jupyter notebooks
├── deployment/                 # Flask web app
├── configs/                    # Configuration files
├── results/                    # Evaluation results
└── logs/                       # Training logs
```

## Setup Instructions

### 1. Clone/Create Project

```bash
mkdir atlas_clothing_classifier
cd atlas_clothing_classifier
```

### 2. Create Virtual Environment

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Setup Kaggle API

1. Go to https://www.kaggle.com/account
2. Create API token (downloads kaggle.json)
3. Place kaggle.json in:
   - Windows: `C:\Users\YourName\.kaggle\`
   - macOS/Linux: `~/.kaggle/`

### 5. Download Dataset

```bash
python src/data/download_dataset.py
```

### 6. Create Data Splits

```bash
python src/data/create_splits.py
```

### 7. Train Model

```bash
python src/training/train.py
```

Training time:
- GPU (NVIDIA GTX 1060+): 3-6 hours
- CPU: 12-24 hours

### 8. Evaluate Model

```bash
python src/evaluation/evaluate.py
```

### 9. Run Web Application

```bash
cd deployment
python app.py
```

Access at: http://localhost:5000

## Usage

### Training

```python
from src.training.train import Trainer

trainer = Trainer('configs/training_config.yaml')
trainer.train()
```

### Inference

```python
from src.inference.predict import predict_image

result = predict_image('path/to/image.jpg')
print(f"Category: {result['category']}")
print(f"Confidence: {result['confidence']}")
```

### Web API

```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/predict
```

## Configuration

Edit `configs/training_config.yaml` to customize:
- Batch size
- Learning rate
- Number of epochs
- Model architecture
- Data augmentation

## Results

Expected performance on Atlas test set:
- **Accuracy**: ~92%
- **Micro F1-Score**: 0.92
- **Macro F1-Score**: 0.88-0.90

## Model Architecture

- **Backbone**: ResNet34 (pretrained on ImageNet)
- **Input**: 224x224 RGB images
- **Output**: 52 classes (clothing categories)
- **Parameters**: ~21M (5M trainable initially)

## Requirements

- Python 3.9-3.11
- CUDA-capable GPU (recommended)
- 20GB free disk space
- 8GB RAM minimum (16GB recommended)

## Troubleshooting

### CUDA Out of Memory
- Reduce batch_size in config (try 32 or 16)
- Reduce num_workers to 2 or 0

### Kaggle API Error
- Check kaggle.json location and permissions
- Verify API token is valid

### Module Not Found
- Ensure virtual environment is activated
- Run: `pip install -r requirements.txt`

## License

This project is for educational purposes.

## Citation

If using the Atlas dataset, please cite:
```
@article{atlas2019,
  title={Atlas: A Dataset and Benchmark for E-commerce Clothing Product Categorization},
  author={Umaashankar, V. and Shanmugam, S.},
  year={2019}
}
```

## Contact

For questions or issues, please open an issue on GitHub.

---

**Project Status**: Production-ready
**Last Updated**: October 2025