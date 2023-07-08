# Flower-Classification
The Flower Classification project is a deep learning project that involves training a model to classify images of flowers into one of 102 categories. The dataset used in this project is the Oxford Flower Dataset, which contains 8,189 images of flowers from 102 different categories. The images are of varying sizes.


![alt text](https://github.com/JananiSBabu/Deep-Flower-Classifier-ResNet50-PyTorch/blob/master/assets/website_img1.png)

The goal of this project is to train a deep learning model that can accurately classify images of flowers into their respective categories. The model is trained using a combination of a pre-trained ResNet50 model and a custom fully connected layer that outputs 102 classes.

The project involves several steps, including loading the dataset, setting up the model, defining the loss function and optimizer, training the model, and evaluating its performance on a validation set. The validation set is used to tune the hyperparameters of the model and prevent overfitting.

The dataset is loaded using a custom PyTorch dataset class called FlowerDataset, which reads in the image files and their corresponding labels from a .mat file. The dataset is split into training, validation, and test sets, and data loaders are created for each set using the DataLoader class.

The model is set up by loading a pre-trained ResNet50 model and replacing its fully connected layer with a new one that outputs 102 classes. The loss function used for this project is the cross-entropy loss, and the optimizer used is stochastic gradient descent (SGD) with a learning rate of 0.001 and momentum of 0.9.

The model is trained for a total of 10 epochs, with the training loss and validation metrics (accuracy, precision, recall, and F1 score) being printed at the end of each epoch. The evaluate function is used to compute the validation metrics, which involves passing the validation data through the trained model and computing the performance metrics based on the predicted and true labels.

After training is complete, the model is evaluated on the test set to obtain a final measure of its performance. The test set is used to evaluate the model's ability to generalize to new, unseen data.

Overall, the Flower Classification project is a good example of how to train a deep learning model for image classification tasks using PyTorch. It demonstrates the importance of data preparation, model selection and setup, hyperparameter tuning, and performance evaluation in the deep learning workflow.

