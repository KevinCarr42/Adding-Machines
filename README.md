# Teaching A Machine To Add

This project examines popular machine learning models in order to gain insights into how these models make predictions. Specifically, how well do these models make predictions given imperfect data and varying sample sizes? This project demonstrates some of the strengths and weaknesses of popular machine learning models by assessing how easily they can be taught to add two numbers.


## Introduction

In this project, the following machine learning models have been investigated:
* Decision Tree
* Naive Bayes
* Linear Regression
* Random Forest
* XGBoost
* TensorFlow

The goal of the project is to gain insight into the predictions from each of these models. Predictions are evaluated based on default models, trained with datasets of varying size and accuracy. 

With each machine learning model, I am asking the following questions: 
* How well will this model predict the addition of 2 numbers? 
* What if the model was given imperfect data?
* How would more data improve predictive ability?

### Rules For This Experiment:
1. No feature engineering or excluding outliers. Otherwise, the problem becomes trivial, and no insights are gained.
2. Default settings only. For the neural network, there is no default setting per se, so I've tried to keep things simple.


## Analysis and Modelling

Models were trained using a variety of datasets. Training datasets were varied in terms of size (number of samples) and accuracy (percent of correct values).

Datasets had the following sizes:
* 100
* 1,000
* 10,000

Dataset accuracy ranged from 50% to 100%. Accuracy determined how many y-values were exactly x1 + x2 in the given dataset. All incorrect data were randomly determined, i.e., they followed a uniform distribution. This method of assigning errors to the training set was chosen to approximate random noise in the data, incorrect inputs, flawed design etc. Other methods could be useful for evaluating these models to gain further insight.

The following dataset accuracies were included in the training dataset:
* 100%
* 99%
* 90%
* 75%
* 50%

For brevity (or at least a bit less verbosity), not all dataset sizes and accuracies were used for every visualisation. Additional analysis are included in the Python notebook file, and were not included in the summary document.

### Scatterplots of Training Set vs Model Predictions

In these plots, the predictions (y-axis) are compared to the correct value (x-axis) in order to visualise errors and predictions. Training datasets are shown in grey, and model predictions are shown in blue.

*Figure 1. Scatterplot of Model Predictions vs Accuracy - 100 Samples of Training Data*
![alt text](https://github.com/KevinCarr42/Teaching-A-Machine-To-Add/blob/main/predictions_100_samples.png)

*Figure 2. Scatterplot of Model Predictions vs Accuracy - 10,000 Samples of Training Data*
![alt text](https://github.com/KevinCarr42/Teaching-A-Machine-To-Add/blob/main/predictions_10k_samples.png)


### Heatmaps of Model Predictions Adding the Number 0 to 10

In these plots, the models predict the addition of every number from 0 to 10 to every number from 0 to 10. One would expect a grid with values from 0 to 20, and a smooth colour gradient from top left to bottom right. 

For "brevity", only models trained with datasets of 10,000 samples were plotted on heatmaps. Additionally, predictions from these models are only shown at three training data accuracies (50%, 90%, and 100%).

*Figure 3. Heatmaps of Predictions - 10,000 Samples of Training Data*
![alt text](https://github.com/KevinCarr42/Teaching-A-Machine-To-Add/blob/main/heatmaps_of_predictions.png)


### Model Performance vs Training Data Size and Accuracy

Model performance was assessed as a function of training data sample size and training data accuracy. The two metrics used to evaluate the performance of the predictions was number of correct answers (rounded to the nearest integer) and mean squared error (MSE).

When assessing performance vs training data sample size, only three sample accuracies were shown (50%, 90%, and 100%).

*Figure 4. Model Performance vs Training Data Sample Size*
![alt text](https://github.com/KevinCarr42/Teaching-A-Machine-To-Add/blob/main/performance_v_size.png)

*Figure 5. Model Performance vs Training Data Accuracy*
![alt text](https://github.com/KevinCarr42/Teaching-A-Machine-To-Add/blob/main/performance_v_accuracy.png)


## Discussion And Conclusions

### Summary of Results:
* Linear regression worked very well as was expected.
* There were very interesting visual patterns created by the predictions of some of the models. This is related to the randomised input data for model training; re-running the calculations leads to different but similar patterns.
* The neural network worked very well, but ended up requiring wider layers than expected. Additionally, dropout layers didn't appear improve performance. Most surprisingly, the neural network trained extremely quickly. More investigation would lead to further insight.
* Completely unsurprisingly, classifiers had very high MSE. However, the classifiers also tended to out-predict other models in terms of total number of correct answers, especially at lower accuracies.

### Potential Future Work:
* Investigate more machine learning models.
* Use a similar project structure to investigate classification.
* Examine linear regression with very large data sets; what is the trade-off between accuracy and size needed to predict correctly?
* Investigate neural networks in more detail. What is the ideal method to train a neural net to add? What trade-offs are involved? Investigate depth vs width, dropout, different optimisation algorithms, etc.

### Other Notes
Sorry for all of the naked Excepts **¯\\\_(ツ)\_/¯**
