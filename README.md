# Teaching A Machine To Add

![thumbnail](https://github.com/KevinCarr42/Teaching-A-Machine-To-Add/blob/main/adding-machines-thumbnail.jpg)

This project examines popular machine learning models in order to gain insights into how these models deal with imperfect data and varying sample sizes. Specifically, this project demonstrates how these models predict the addition of two numbers.

## Introduction

In this project, six popular machine learning algorithms have been investigated. Wikipedia articles for each model have been linked below:
* [Decision Tree](https://en.wikipedia.org/wiki/Decision_tree)
* [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
* [Linear Regression](https://en.wikipedia.org/wiki/Linear_regression)
* [Random Forest](https://en.wikipedia.org/wiki/Random_forest)
* [XGBoost](https://en.wikipedia.org/wiki/XGBoost)
* [TensorFlow](https://en.wikipedia.org/wiki/TensorFlow)

It is important to note that goal of this project is not to accurately predict the addition of two numbers. Rather, the goal of this project is to gain insights about *how* each model makes those predictions.

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

For brevity (or at least a bit less verbosity), not all dataset sizes and accuracies were used for every visualisation. Additional analyses are included in the Python notebook file, and were not included in the summary document.

### Scatterplots of Training Set vs Model Predictions

In these plots, the predictions (y-axis) are compared to the correct value (x-axis) in order to visualise errors and predictions. Training datasets are shown in grey, and model predictions are shown in blue.

*Figure 1. Scatterplot of Model Predictions vs Accuracy - 100 Samples of Training Data*
![fig1](https://github.com/KevinCarr42/Teaching-A-Machine-To-Add/blob/main/predictions_100_samples.png)

*Figure 2. Scatterplot of Model Predictions vs Accuracy - 10,000 Samples of Training Data*
![fig2](https://github.com/KevinCarr42/Teaching-A-Machine-To-Add/blob/main/predictions_10k_samples.png)

All models follow some approximation of the basic slope of y<sub>predicted</sub> = y<sub>correct</sub>. However, each model has a unique distribution of predictions, with the exception of Decision Trees and Random Forests, whose similarity is due to Random Forests being an ensemble of Decision Trees.

An interesting note, the decrease in slope observed in the linear regression model is also observed in the TensorFlow model. For the linear regression model, this change in slope is due to the average value of the predictions being "pulled towards zero" by the random noise from the training set (i.e., the "errors" in training data, which are uniformly distributed about 0). It is unknown whether there is a similar mechanism causing this effect in the TensorFlow model.

### Heatmaps of Model Predictions Adding the Number 0 to 10

In these plots, the models predict the addition of every number from 0 to 10 to every number from 0 to 10. One would expect a grid with values from 0 to 20, and a smooth colour gradient from top left to bottom right. 

For "brevity", only models trained with datasets of 10,000 samples were plotted on these heatmaps. Additionally, predictions from these models are only shown at three training data accuracies (50%, 90%, and 100%).

*Figure 3. Heatmaps of Predictions - 10,000 Samples of Training Data*
![fig3](https://github.com/KevinCarr42/Teaching-A-Machine-To-Add/blob/main/heatmaps_of_predictions.png)

As seen in these heatmaps, each of the models makes prediction errors in distinctive patterns, with approximations becoming better as accuracy in the training data improves. In the linear regression and TensorFlow models, predictions are "pulled towards zero" for lower-accuracy training sets as noted in the Scatterplot discussion.

Patterns in errors in prediction from each of the other models (Decision Tree, Naive Bayes, Random Forest, and XGBoost) do not follow easily predictable patterns.

### Model Performance vs Training Data Size and Accuracy

Model performance was assessed as a function of training data sample size and training data accuracy. The two metrics used to evaluate the performance of the predictions was number of correct answers (rounded to the nearest integer) and mean squared error (MSE).

When assessing performance vs training data sample size, only three sample accuracies were shown (50%, 90%, and 100%).

*Figure 4. Model Performance vs Training Data Sample Size*
![fig4](https://github.com/KevinCarr42/Teaching-A-Machine-To-Add/blob/main/performance_v_size.png)

Increased training data size generally decreases the mean squared error (MSE) and increases the number of correct guesses for each model.

*Figure 5. Model Performance vs Training Data Accuracy*
![fig5](https://github.com/KevinCarr42/Teaching-A-Machine-To-Add/blob/main/performance_v_accuracy.png)

Similarly, increased training data accuracy generally decreases the mean squared error (MSE) and increases the number of correct guesses for each model.

Linear regression appears to be the most accurate predictive model, due to the lowest MSE under all dataset sizes and accuracy levels (with 1 potentially anomalous exception, Naive Bayes outperforms at 10000 samples of training data at 50% accuracy). With 100% accuracy in training data, the linear regression model outputs perfect predictions, guessing 100% of answers with a MSE of 0. However, when training data accuracy decreases, the number of correct guesses from the linear regression model starts losing to Decision Tree, Random Forest, and XGBoost models (although Decision Tree and Random forest models have very large MSE at these training data accuracy levels).

## Discussion And Conclusions

### Summary of Results:
* Linear regression outperformed most models under most conditions.
* Classifier and regressor implementations of decision tree models were included. Both performed similarly, even though adding 2 numbers should be considered a regression task.
* There were very interesting visual patterns created by the predictions of some of the models. This is related to the randomised input data and model training; re-running the calculations leads to different but similar patterns.
* The TensorFlow neural network also worked very well in most conditions.
* Decision Tree and Random Forest models had very high MSE. However, these models also tended to out-predict other models in terms of total number of correct answers, especially at lower accuracies.

### Potential Future Work:
* Investigate more machine learning models.
* Investigate approaches and trade-offs required for these models to make predictions correctly / optimally.
* Investigate neural networks in greater detail.
* Investigate other basic mathematical functions (e.g., multiplication, exponentiation, inequalities).
* Use a similar project approach to investigate a classification problem.
