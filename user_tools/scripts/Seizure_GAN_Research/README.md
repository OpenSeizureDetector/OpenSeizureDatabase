# 🌀 GAN-Based Acceleration Data Generation

## Introduction  
This project uses **Generative Adversarial Networks (GANs)** to generate synthetic acceleration data from a dataset of epileptic seizure recordings captured using an accelerometer.

### Key Features  
- Generates realistic acceleration data mimicking real-world seizure recordings.  
- Utilizes advanced normalization and denormalization techniques for data preprocessing and postprocessing.  
- Enables detailed comparison between real and synthetic data through visualization.  

## 📊 Dataset  

### Source  
The dataset is sourced from the **[Open Seizure Database](https://www.epilepsydatabase.org)**. It contains 1D acceleration data from epileptic seizures.  

### Data Description  
- **Type**: 1D vector signal as vector magnitude (rawData)  
- **Format**: Time-series data.  
- **Use Case**: Train the GAN to generate synthetic acceleration data mimicking real seizure recordings.  

## 🏗️ GAN Architecture  

### Overview  
The Generative Adversarial Network (GAN) consists of two key components:  
1. **Generator (Gen)**: Creates synthetic acceleration data.  
2. **Discriminator (Desc)**: Differentiates between real and fake acceleration data.  

### Generator  
🔧 **Functionality**:  
- Takes random noise (latent points) as input.  
- Outputs synthetic acceleration data in the same format as the real data.  

💡 **Key Features**:  
- Fully connected and convolutional layers for generating structured outputs.  
- Trained to produce data that resembles the real dataset closely.  

### Discriminator  
🔍 **Functionality**:  
- Accepts both real and fake data as input.  
- Predicts whether the input is real (from the dataset) or fake (from the generator).  

## ⚙️ Installation  

### Prerequisites  
Before running the code, ensure the following tools and libraries are installed:  
- **Python** (Version 3.8 or later)  
- **TensorFlow** (Version 2.x)  
- **NumPy**  
- **Matplotlib**  
- **scikit-learn**  

### Installation Steps  
1. Clone this repository:  
```bash
git clone https://github.com/jpordoy/AMBER.git
cd GAN-Acceleration-Data
```

```python
python -m venv env  
source env/bin/activate  # On Windows: env\\Scripts\\activate
```

### Adapted Code
- This project includes adaptations from code originally developed by Bjorn in the Kaggle notebook "GAN on ECG", which uses a 1D CNN-based GAN to generate synthetic 12-lead ECGs from the PTB-XL Dataset.

