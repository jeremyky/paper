# Installation Guide

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)
- Git

## Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/jeremyky/remote-viewing-experiment.git
   cd remote-viewing-experiment
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Required Models**
   The first run will automatically download the ResNet-50 model.

## Configuration

1. **Data Directory Setup**
   ```bash
   mkdir -p data/raw/images
   # Copy your images to data/raw/images/
   ```

2. **Environment Variables**
   Create a `.env` file:
   ```
   DATA_DIR=data/raw/images
   OUTPUT_DIR=experiments
   ```

## Verification

Run the test script to verify installation:
```bash
python scripts/test_real_data.py
```
