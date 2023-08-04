# CV Carvana Masking Challenge YPAI04 Nurulhusna
 This is an exercise to create a model for a semantic segmentation problem known as the Carvana Masking Challenge by Nurulhusna (YPAI04)

This project aims to solve the Carvana Masking Challenge, which is a semantic segmentation problem for images of cars. The goal is to build a U-net model to perform image segmentation and accurately mask the cars in the images.

## Dataset

You can obtain the dataset for the Carvana Masking Challenge from the following link:

[Carvana Dataset](https://www.kaggle.com/c/carvana-image-masking-challenge/data)

Please unzip the dataset and place it in the same folder as your code file.

## Prerequisites

To run and complete the Carvana Masking Challenge project, you will need the following requirements:

1. Python (>=3.6): The project is implemented in Python

2. TensorFlow (>=2.0): TensorFlow is a deep learning framework, and it's used for building and training the U-net model.

3. TensorFlow Datasets (tfds): This library is used to load and manage datasets efficiently, including the Carvana Masking dataset.

4. TensorFlow Examples (tf-examples): The `pix2pix` module from TensorFlow Examples is used for constructing the upsampling path in the U-net model.

5. NumPy: NumPy is used for handling numerical operations and creating NumPy arrays.

6. OpenCV (cv2): OpenCV is used for image processing tasks, such as reading and resizing images.

7. Matplotlib: Matplotlib is used for displaying images and visualizations during the project.

8. Scikit-learn: Scikit-learn is used for data preprocessing, specifically for splitting the dataset into training and testing sets.

9. Streamlit (for deployment, mentioned in the original project): If you plan to deploy the model using Streamlit, you'll need Streamlit library for building the interactive web app.


## Instructions

1. Load the images and masks from the dataset. 

2. Prepare the dataset for model training by converting the loaded data into NumPy arrays and performing data preprocessing steps such as expanding mask dimensions, converting mask values to 0 and 1, and normalizing the image pixel values.

3. Split the dataset into training and testing sets using `train_test_split` from `sklearn.model_selection`.

4. Convert the NumPy arrays into TensorFlow tensors and combine the features and labels to form a zip dataset.

5. Define the hyperparameters for the TensorFlow dataset, including batch size and buffer size.

6. Create a data augmentation layer through subclassing to augment the training data.

7. Build the dataset for training and testing by performing data caching, shuffling, batching, repeating, and mapping the data augmentation.

8. Define a U-net model for image segmentation. This involves using a pre-trained MobileNetV2 model as the feature extractor and constructing the upsampling path.

9. Compile the model using the appropriate loss function and optimizer.

10. Create a function to display prediction results during model training.

11. Train the U-net model using the training dataset and validate it using the testing dataset. Monitor the progress using TensorBoard.

12. Deploy the model to make predictions on new data. Display sample predictions on the test dataset.


## Acknowledgments
The datasets used in this project are sourced from Kaggle. Credit goes to the original author:
 [Carvana Dataset](https://www.kaggle.com/c/carvana-image-masking-challenge/data)

## License
This project is licensed under the [MIT License](LICENSE).
