# DeepvisionNet64x64-CNN

A simple and effective **Convolutional Neural Network (CNN)** model for image classification with 64x64 input size.  
Designed for binary classification with data augmentation, training, and fine-tuning capabilities.

---

## Model Performance

- Final Training Accuracy: 99%
- Final Validation Accuracy: 97%
- These accuracy values represent the model’s performance on the training and validation datasets, respectively.


## Features

- Input images of size 64x64 with 3 color channels (RGB)  
- Architecture includes 3 convolutional layers with MaxPooling and Dropout to reduce overfitting  
- Uses `ImageDataGenerator` for data augmentation  
- Supports fine-tuning on new datasets without training from scratch  
- Model saving and loading in `.keras` format  
- Training history plots for accuracy and loss  

---

## Installation and Setup

Make sure you have Python and the required packages installed:

```bash
pip install tensorflow matplotlib

---

## Directory Structure

│
├── Images/                # Initial dataset for training (2 classes)
│   ├── class1/
│   └── class2/
│
├── DOGS/                  # New dataset for fine-tuning (2 classes)
│   ├── class1/
│   └── class2/
│
├── CNN_model.keras        # Saved model after initial training
├── CNN_model_finetuned.keras  # Saved model after fine-tuning
├── model_training.py      # Code for training and fine-tuning the model
└── README.
---- 
 ## Training the Initial Model
python Main_CNN_without_fine_tune.py
Uses data augmentation to improve model robustness
Saves the trained model as CNN_model.keras
---


## Fine-tuning the Model
After initial training, the model can be fine-tuned on new data located in the DOGS folder:
Freezes earlier layers to preserve learned features
Trains only the last layers for 10 epochs
Saves the fine-tuned model as CNN_model_finetuned.keras
------

## Using the Saved Model
Load the fine-tuned model for inference or further training:

import tensorflow as tf

model = tf.keras.models.load_model('CNN_model_finetuned.keras')

----

## Dataset
The training and validation images used for this project can be downloaded from the following public folder:

[Download dataset here]
(https://drive.google.com/drive/folders/1i_opj382YPOEHYah98hYY1eDla2XegbS?usp=sharing)
(https://drive.google.com/drive/folders/1_vfb4duUJuvi63GQU9JNexNxREZR_RXL?usp=sharing)

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
Feel free to reach out if you have any questions or suggestions!
mohammad.rz.samadi@gmail.com

Thanks for checking out this project! 🙌













