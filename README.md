# Seismic Data Analysis

<img width="992" alt="image" src="https://github.com/user-attachments/assets/1f737005-7e09-44e2-a78d-f42264957da7" />


## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact/Acknowledgments](#contactacknowledgments)

## Overview

This project implements a deep semi-supervised learning (DSSL) approach for seismic data analysis, specifically focused on estimating acoustic impedance using pre-stack seismic data. The project addresses a critical challenge in reservoir characterization: the scarcity of labeled data despite the abundance of unlabeled seismic data.

Seismic data preparation is an exceptionally complex and challenging process that involves sophisticated techniques for squeezing and stretching the data. This preprocessing is crucial for aligning seismic traces with well log data and ensuring accurate analysis. In adition to this challegenging data preparation, the project handles other pre-processing functions that normalize, interpolate, and transform raw seismic data into formats suitable for deep learning models.

The solution leverages recurrent neural networks in a semi-supervised learning framework to improve the performance of supervised learning by utilizing large amounts of unlabeled data. This approach has proven particularly effective in domains where labeled data is expensive or difficult to obtain, such as oil exploration and reservoir characterization.

## Dataset

The dataset used in this project consists of:

1. **3D Pre-stack Seismic Data**: Organized into gathers, also known as common depth points (CDP). These gathers contain multiple seismic traces that provide rich information about subsurface structures.

2. **Well Log Data**: Contains acoustic impedance measurements from five borehole logs, which serve as the labeled data for training and validation.

The preprocessing pipeline includes:
- Extraction of seismic data around well locations
- Normalization and interpolation of seismic and well log data
- Reduction of samples in traces to optimize processing
- Windowing of data per CDP for effective training

Data is stored in TensorFlow's TFRecord format after preprocessing, with separate datasets for training, testing, and unlabeled data.

## Installation

To set up the project environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/seismic-data-analysis.git
cd seismic-data-analysis

# Install dependencies
pip install -r requirements.txt
```

Required dependencies include:
- TensorFlow (2.x)
- NumPy
- Pandas
- SciPy
- scikit-learn
- segysak (for SEG-Y file handling)

Python 3.7+ is recommended for optimal compatibility.

## Usage

### Data Preprocessing

To preprocess raw seismic and well log data:

```python
from seismic.npseismic import ringOfData

# Process data for a specific well
ringOfData(
    _well_name="well-1",
    _path_sgy_smaller_cube="path/to/seismic_cube.sgy",
    _syntheticdir="path/to/synthetic_well_log.txt",
    _dirToSave="processed_data/",
    _well_iline=123,
    _well_xline=456,
    _startTime=1000,
    _stopTime=2000,
    _root="project_root/"
)
```

### Running Inference

To estimate acoustic impedance from preprocessed seismic data:

```python
# Load the trained model
from training.semi_supervised.fine_tuning import load_model
model = load_model("path/to/model")

# Run inference on new data
predictions = model.predict(seismic_data)
```

## Training

The project implements a two-stage semi-supervised learning approach:

### 1. Pre-training Stage

```python
# Run pre-training on unlabeled data
python training/semi-supervised/pre-training/pre-training_on_server_1.py
```

This stage uses unlabeled data to train LSTM autoencoders in a greedy layer-wise fashion, helping the model learn meaningful representations from the abundant unlabeled seismic data. Pre-training was performed using two GPUs to enable parallel computation and reduce training duration.

### 2. Fine-tuning Stage

```python
# Fine-tune the pre-trained model with labeled data
python training/semi-supervised/fine_tuning/fine_tuning.py
```

This stage uses the limited labeled data to fine-tune the pre-trained model for the specific task of acoustic impedance estimation.

The training process typically takes several minutes on a GPU-enabled machine, with the pre-training stage being the most computationally intensive.

## Results

The deep semi-supervised learning approach demonstrates superior performance compared to purely supervised methods when working with limited labeled data:

- **Improved Accuracy**: The DSSL model achieves better acoustic impedance estimation both at well locations and outside them.
- **Better Generalization**: The model shows enhanced ability to generalize to unseen data.
- **Reduced Data Requirements**: Achieves good performance with significantly less labeled data than traditional supervised approaches.

The results validate that employing large amounts of unlabeled data can significantly improve seismic data interpretation systems, particularly in scenarios where labeled data is scarce or expensive to obtain.

## Project Structure

```
seismic-data-analysis/
├── pre-processing/                # Data preprocessing scripts
│   ├── gathering_data_of_all_wells_as_labeled_test_and_unlabeled.py
│   ├── gathering_data_of_well-n.py
│   └── saving_dataset_as_tfrecords_well-n.py
├── seismic/                       # Core seismic processing package
│   └── npseismic.py               # Main functions for seismic data processing
├── training/                      # Training scripts and models
│   ├── semi-supervised/           # Semi-supervised learning implementation
│   │   ├── pre-training/          # Pre-training on unlabeled data
│   │   │   ├── pre-training_on_server_1.py
│   │   │   └── pre-training_on_server_2.py
│   │   └── fine_tuning/           # Fine-tuning with labeled data
│   │       └── fine_tuning.py
│   └── supervised/                # Supervised learning implementation
├── hierarchy_of_functions.txt     # Documentation of function dependencies
└── README.md                      # Project overview
```

## Contributing

Contributions to this project are welcome. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please ensure your code follows the project's coding style and includes appropriate tests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact/Acknowledgments

This project was developed by Edgar Ek-Chacón, Erik Molino-Minero-Re, Paul Erick Méndez-Monroy, Antonio Neme, and Hector Ángeles-Hernández as part of research at the Universidad Nacional Autónoma de México (UNAM).

For questions or collaborations, please contact:
- Edgar Ek-Chacón: ekchacon89@gmail.com
- Erik Molino-Minero-Re: erik.molino@iimas.unam.mx

The research was published in Applied Sciences journal under the title "Semi-Supervised Training for (Pre-Stack) Seismic Data Analysis."
