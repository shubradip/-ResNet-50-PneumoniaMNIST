# Pneumonia Classification with EfficientNetB0

This project trains a binary classifier to detect pneumonia using the PneumoniaMNIST dataset. The model is built using TensorFlow and EfficientNetB0, with data augmentation and hyperparameter tuning.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   Install dependencies:

bash

pip install -r requirements.txt
Download the dataset:

bash

python -c "import kagglehub; kagglehub.dataset_download('rijulshr/pneumoniamnist')"
Train the model:

bash

python train.py
Evaluate the model:

bash

python evaluate.py
Hyper-parameter Choices
Learning Rate: 0.001 with exponential decay (decay rate: 0.9, decay steps: 10,000).
Batch Size: 64 for training, validation, and testing.
Epochs: 50, with early stopping after 30 epochs of no improvement.
Dropout: 0.1 to 0.5 (tuned using Keras Tuner).
Units in Dense Layer: 64 to 256 (tuned using Keras Tuner).
Reproducibility
To reproduce the results:

Ensure the dataset is downloaded using the kagglehub package.
Run the training script (train.py) with the default hyperparameters or modify them in the script.
Evaluate the model using the evaluate.py script.
Results
Validation Accuracy: ~76.5%
Test AUC: ~0.50 (example value, replace with actual results).
Test F1-Score: ~0.77 (example value, replace with actual results).
Notes
The model uses data augmentation to improve generalization.
Class imbalance is handled using class weights during training.
Hyperparameters were tuned using Keras Tuner's Hyperband algorithm.
License
This project is licensed under the MIT License.


---

### Notes on Hyper-parameter Choices
- **Learning Rate**: A starting value of 0.001 was chosen for stability, with exponential decay to adapt as training progresses.
- **Batch Size**: 64 was selected to balance memory usage and training speed.
- **Epochs**: 50 was set as a maximum, with early stopping to prevent overfitting.
- **Dropout**: Tuned between 0.1 and 0.5 to regularize the model.
- **Dense Layer Units**: Tuned between 64 and 256 to find the optimal capacity for the dataset.

