# Facial Emotion Recognition Project

This project aims to recognize facial emotions using various convolutional neural network (CNN) architectures, including a normal CNN network, ResNet, and MobileNet. The best results were achieved using MobileNet, with an accuracy of 67.54%. (CNN -> acc=53.16%, resnet -> acc=65.85%)

## Introduction

Facial emotion recognition is a task in computer vision that involves detecting and categorizing facial expressions into different emotional states, such as happiness, sadness, anger, etc. This project provides a solution for facial emotion recognition using deep learning techniques.

Course: EECS 841, Computer vision @ The University of Kansas



## Usage

To test the best model achieved in this project with your own dataset, follow these steps:

1. **Clone the Repository**: Clone this repository to your local machine using the following command:

2. **Prepare Your Dataset**: Place your test dataset in the `dataset_test_II` directory with your own dataset. Ensure that your dataset follows the same directory structure and file naming conventions.

3. **Update Test Script**: Open the `test_phase_II.py` file and update the `test_data_dir` variable at the beginning of the script to point to the directory containing your test dataset.

4. **Install Dependencies**: Make sure you have all the necessary dependencies installed. You can install them using pip:

The project uses the following libraries:
- `numpy` (version 1.23.5)
- `pandas` (version 1.4.3)
- `torch` (version 1.13.1)
- `tqdm` (version 4.64.1)
- `scikit-learn` (version 1.1.3)
- `torchvision` (version 0.15.2a0)
- `matplotlib` (version 3.7.2)

You can install these dependencies using pip:

5. **Run the Test Script**: Execute the `test_phase_II.py` script to test the best model with your dataset:


6. **View Results**: After running the script, the results will be generated, including accuracy and confusion matrix. You can analyze the results to evaluate the performance of the model on your dataset.

## Directory Structure

- **dataset/**: Contains the original dataset provided with the project. Replace with your own dataset.
- **confusion_matrix/**: Contains the confusion matrix of each epoch for each method under three folders with methods' names.
- **results/**: Contains the results of each epoch for each method under three folders with methods' names.
- **dataset_test_II/**: Place with your own testing dataset.
- **checkpoint/**: Contains checkpoints of trained models.
- **test_phase_II.py**: Python script to test the best model with your test dataset placed at dataset_test_II folder.
- **utils.py**: util functions for data preprocessing and evaluation calculations.
- **CNN.py**: training and testing with CNN method.
- **MOBILENET.py**: training and testing with MOBILENET method.
- **RESNET.py**: training and testing with RESNET method.
- **requirements.txt**: List of Python dependencies required for the project.




## License

This project is licensed under the MIT License.
