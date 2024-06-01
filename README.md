# Big Cats Image Classification Project

## Table of Contents
1. Project Goal and Motivation
2. Data Collection
3. Modeling
4. Training
5. Interpretation and Validation
6. Repository Structure
7. How to Run

## Project Goal and Motivation
The primary goal of this project is to develop a robust machine learning model capable of accurately classifying images of various big cat species. This model aims to distinguish between different species of big cats such as lions, tigers, leopards, and others. The classification model will be based on a pre-trained convolutional neural network, specifically MobileNetV2, which will be fine-tuned to enhance its performance on this specific task.
Big cats are crucial indicators of ecosystem health and face significant threats from habitat loss and poaching. Accurate identification and classification of these species play a vital role in conservation efforts. Automating species identification aids in monitoring and protecting endangered big cats, facilitating more efficient conservation strategies. 

## Data Collection
The data which is being used for this project has been collected on Kaggle.com containing images of 10 different big cat species: https://www.kaggle.com/datasets/gpiosenka/cats-in-the-wild-image-classification/data
The dataset is structured into three main directories: train, test, and valid, each containing subdirectories for each species. The train directory contains images used to train the model. This data is crucial for teaching the model to recognize patterns and features specific to each big cat species.
The valid directory holds images used to validate the model's performance during the training process. This helps in tuning the model's hyperparameters and preventing overfitting.
The test directory includes images used for the final evaluation of the model. This data is not seen by the model during training, providing an unbiased assessment of its performance.

## Modeling
### Preprocessing
The images are preprocessed and augmented using TensorFlow's ImageDataGenerator to enhance the model's generalization ability. The preprocessing function from MobileNetV2 is applied, and various augmentations such as rotation, zoom, width shift, height shift, shear, and horizontal flip are used to create a more robust training dataset.
### Model Architecture
The core of the model is based on MobileNetV2, a pre-trained convolutional neural network known for its efficiency and accuracy. The architecture is modified to suit the specific needs of this image classification task. 
New fully connected layers are added to adapt the model to classify the 10 big cat species.
### Compilation
The model is compiled using the Adam optimizer, which is known for its efficiency and effectiveness in training deep learning models. The loss function used is categorical cross-entropy, appropriate for multi-class classification problems. The model's performance is tracked using accuracy as the evaluation metric.
### Training
The model is trained on the training dataset with real-time data augmentation. The training process includes:
- Early Stopping: To prevent overfitting, early stopping is implemented. The training is monitored for validation loss, and if the loss does not improve for 5 consecutive epochs, the training stops and the best weights are restored.
- Validation: During training, the model's performance is validated using the valid dataset. This helps in tuning the model and preventing overfitting.

## Interpretation and Validation
For the final validation, the test dataset, which was not seen by the model during training, was used to assess its performance. This provides an unbiased evaluation of the model's ability to generalize to new, unseen data.
The results of the final validation with the test dataset are as follows:
- Accuracy:  The model achieved an accuracy of 96.00% on the test set. This high accuracy indicates that the model can reliably classify images of big cats into their respective species. 
- Classification Report: A detailed classification report was generated, which includes precision, recall, and F1-score for each class. These metrics provide insights into the model's performance for each individual species, highlighting its strengths and areas for improvement.
|

- Confusion Matrix:  A normalized confusion matrix was created to visualize the model's performance across all classes. The confusion matrix helps identify which species are being correctly classified and which ones are causing confusion for the model.
- Within the Confusion Matrix there have been only 2 values les### Classification Report

The classification report provides detailed performance metrics for each species in the test dataset.

| Species            | Precision | Recall | F1-Score | Support |
|--------------------|-----------|--------|----------|---------|
| AFRICAN LEOPARD    | 1.00      | 0.80   | 0.89     | 5       |
| CARACAL            | 1.00      | 1.00   | 1.00     | 5       |
| CHEETAH            | 0.83      | 1.00   | 0.91     | 5       |
| CLOUDED LEOPARD    | 1.00      | 0.80   | 0.89     | 5       |
| JAGUAR             | 0.83      | 1.00   | 0.91     | 5       |
| LIONS              | 1.00      | 1.00   | 1.00     | 5       |
| OCELOT             | 1.00      | 1.00   | 1.00     | 5       |
| PUMA               | 1.00      | 1.00   | 1.00     | 5       |
| SNOW LEOPARD       | 1.00      | 1.00   | 1.00     | 5       |
| TIGER              | 1.00      | 1.00   | 1.00     | 5       |
| **Accuracy**       |           |        | **0.96** | **50**  |
| **Macro avg**      | 0.97      | 0.96   | 0.96     | 50      |
| **Weighted avg**   | 0.97      | 0.96   | 0.96     | 50      |

- A normalized confusion matrix was created to visualize the model's performance across all classes. The confusion matrix helps identify which species are being correctly classified and which ones are causing confusion for the model. There were only 2 values less than 1. And these values were 0.8 - still high based on the fact that there are some big cats in the dataset which seem to have very strong similarities. The two Categories with the 0.8 value were the African Leopard and the Clouded Leopard. Considering that within the Data Set the African Leopard, Caracal, Cheetah, Clouded Leopard, Jaguar and the Snow Leopard have strong similarities, this is a value to be happy of.

### Conclusion
The final validation demonstrates that the model performs exceptionally well in classifying big cat species, achieving high accuracy and strong performance metrics across various species. The detailed analysis using the classification report and confusion matrix provides a comprehensive understanding of the model's strengths and areas for potential improvement. This robust validation process ensures that the model is not only accurate but also reliable for real-world applications.

## Repository Structure
The repository is organized to ensure clarity and ease of use, allowing for straightforward reproduction of the project results. Below is a detailed description of the repository structure:
- -FinalMLProjectAjrizple
  - -- test
  - -- train
  - -- valid
  - -- model.ipynb
  - -- requirements.txt
 
## How to run
To reproduce the results of this project, follow these steps:
1. Clone the Repository
2. Set Up the Environment: Ensure you have Python 3.12.1 installed on your machine. It's recommended to use a virtual environment to manage dependencies.
3. Install Dependencies: Install the required dependencies using pip and the requirements.txt file provided in the repository: pip install -r requirements.txt
5. Run the Notebook: Execute all the cells in the notebook to run the entire project.
6. As you execute the cells, you will see various outputs, including training progress, accuracy, loss graphs, classification reports, and confusion matrices. Ensure that your outputs match the expected results as described in the README.md file. 
### Important
Keep in mind that the three folders with the pictures have to be placed within the repository for the notebook to work properly!
