# GAN-Based Cell and Slide Generation of Cancer Cells for SIPaKMED

This project implements a Generative Adversarial Network (GAN) to generate synthetic cell images and medical slides. It is designed to augment datasets by generating high-quality artificial images based on a real dataset of medical slides. The GAN is trained on a dataset of labeled cell images with a 60:20:20 split for training, validation, and testing.

## Features
- **Image Preprocessing**: Loads, resizes, and processes cell images from a directory structure, ensuring all images are standardized to 64x64x3 dimensions.
- **GAN Architecture**:
  - **Generator**: Generates realistic 64x64x3 cell images from random noise using a deep neural network.
  - **Discriminator**: Evaluates images, distinguishing between real and generated ones, using a convolutional network.
- **Training**:
  - Binary Cross-Entropy loss functions for both generator and discriminator.
  - Alternating training between generator and discriminator to improve image generation over time.
- **60:20:20 Data Split**:
  - Training: 60% of the images.
  - Validation: 20% of the images.
  - Test: 20% of the images.
- **Visualization**: Displays generated images after each epoch for qualitative evaluation of the training process.

## Table of Contents
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Results](#results)
- [Technologies](#technologies)
- [Future Improvements](#future-improvements)

## Briefly What is SIPaKMED
This project aims to develop a Vanilla Generative Adversarial Network (GAN) to generate synthetic cervical cell images using the SIPaKmed dataset. The generated images are intended to enhance diagnostic training resources for cervical cancer detection, providing a valuable tool for researchers and medical professionals.

## Background
SIPaKmed is a publicly available dataset that contains cervical cell images, classified into various categories. The goal of this project is to leverage deep learning techniques to generate realistic synthetic images, which can augment the existing dataset and assist in training models for cervical cancer diagnosis.

## Project Description
- Developed a Vanilla GAN model to generate synthetic cervical cell images.
- Preprocessed and organized the SIPaKmed dataset to ensure high-quality input images.
- Optimized hyperparameters using Optuna to enhance model performance.
- Validated the model on unseen test data to assess image realism and diversity.

## Installation
To run this project, you need to have Python 3.6 or higher installed along with the following libraries:

## bash
`pip install torch torchvision matplotlib opencv-python optuna`
Usage
Clone the repository:

## bash
Copy code
`git clone https://github.com/yourusername/synthetic-cell-image-generation.git`
cd synthetic-cell-image-generation
Run the training script:

##bash
Copy code
`python train.py`
Generated images will be saved in the output/ directory.

## Results
The Vanilla GAN successfully generated synthetic cervical cell images that exhibit realistic characteristics. The generated images were evaluated based on their visual fidelity and potential utility in medical diagnostics.


## License
This project is licensed under the MIT License. See the LICENSE file for more information.

## Acknowledgments
Dr. Prof. Nibarana Das for guidance and support throughout the project.
Contributors and maintainers of the SIPaKmed dataset.

### Notes:
- Replace `yourusername` in the clone command with your actual GitHub username.
- Make sure to update the `output/sample_output.png` path to point to an actual output image in your repository, or remove that section if you prefer.
- Feel free to customize the content further based on your project specifics!
