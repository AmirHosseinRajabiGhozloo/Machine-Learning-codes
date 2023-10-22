# Python-Codes
This repository contains a collection of programming projects implemented in Python. These projects cover different subjects, with a focus on image segmentation and computer vision.

# 1. COVID-Detection

This code demonstrates the process of building an image classification model using transfer learning with the InceptionResNetV2 architecture.
It preprocesses the training data, constructs the model, and compiles it for training. 
The code provides a foundation for further training and evaluating the model on the given COVID-19 and Non-COVID-19 image dataset.

# 2. Cancer-Detection

This code segment focuses on cancer image segmentation using the U-Net model.
It preprocesses the cancer-related data, defines the specific architecture for cancer segmentation, provides IOU metrics for evaluating model performance, and sets up the model for training on cancer datasets.

# 3. HyperParameter-Tuning

This code provides a framework for training and optimizing a 2D U-Net model for rock image segmentation.
By leveraging custom loss functions, evaluation metrics, and hyperparameter tuning, the aim is to achieve accurate and precise segmentation of rocks in images, which can have applications in various domains such as geological analysis, environmental monitoring, and resource exploration.
Hyperparameter tuning have been done with Ray Tune. It specifies the search space for the learning rate and sets the stopping criteria.
The tuner fits the model using the defined configuration and returns the best hyperparameters found.

# 4. MNIST-Sequential

This code demonstrates the construction and training of a neural network model for handwritten digit classification using the MNIST dataset.
By leveraging the TensorFlow and Keras libraries, the code showcases a straightforward implementation of a deep learning model, providing insights into the training progress and performance through visualizations.
