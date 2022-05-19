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


## Results

### Visualisation of Model Predictions

#### Scatterplots of Training Set vs Model Predictions



#### Heatmaps of Model Predictions Adding the Number 0 to 10



#### Model Performance vs Training Data Size and Accuracy


## Discussion And Conclusions

### Summary of Results:
* Linear regression worked very well as was expected.
* There were very interesting visual patterns created by the predictions of some of the models. This is related to the randomised input data for model training; re-running the calculations leads to different but similar patterns.
* The neural network worked very well, but ended up requiring wider layers than expected. Additionally, dropout layers didn't appear improve performance. Most surprisingly, the neural network trained extremely quickly. More investigation would lead to further insight.
* Completely unsurprisingly, classifiers are not great at adding.

### Potential Future Work:
* Investigate more machine learning models.
* Use a similar project structure to investigate classification.
* Examine linear regression with very large data sets; what is the tradeoff between accuracy and size needed to predict correctly?
* Investigate neural networks in more detail. What is the ideal method to train a neural net to add? What trade-offs are involved? Investigate depth vs width, dropout, different optimisation algorithms, etc.

### Other Notes
Sorry for all of the naked Excepts **¯\\\_(ツ)\_/¯**
