# Quantifying Uncertainty of DNN Hyperparameter Optimization using a First Order Type

Hyperparameter optimization is a difficult problem in developing deep learning applications. Recently, random search based strategies have been proven efficient for optimizing hyperparameters. However, programmers can not overtly represent uncertainty of the chosen hyperparameter values and accuracy of the model while performing a random search. In this project, we utilize a first order type Uncertain<T> to approximate the distributions of the hyperparame- ters so that programmers can pick values with certain confidence. This type helps us to represent uncertainty of random hyperparameters and allows us to easily propagate this uncertainty through computations, perform statistical tests on dis- tributions without resorting to complicated statistical concepts, and determine uncertain hyperparameter value in required significance level.

To the best of our knowledge, there has not been any attempt to introduce the probabilistic programming concept in DNN hyperparameter optimization. The contributions of this project are as follows. First, we have implemented the first order type Uncertain<T> to hold the distribution of loss values over the randomly chosen hyperparameters. The main goal is to help programmers overtly represent uncertainty in chosen hyperparameters and make conditional statements using that. By using this type, we define algebra over random variables so that the un- certainty of the hyperparameters can propagate through the calculations and pro- vide convergence speed and increase in accuracy. Second, our method performs significantly better than the random search method which is used by most of the DNN libraries. Our result shows that while 62% of the random search trials fall below the accuracy threshold, only 23% time our method fall below the threshold.


In this project, our goal is to aid the deep learning programmers to quantify uncer- tainty in the model hyperparameters and make an informed decision while initializing hyperparameters. A problem with the random search is that it doesn’t take uncertainty of random hyperparameters into account while picking a best value which may adversely impact models when trained on the different distribution of the input domain. To that end, we have leveraged a first order type Uncertain<T> to represent the uncertainty in the random hyperparameters and choose best value by performing statistical tests on the distribution. The main contributions of the project are:

1. We have introduced the probabilistic programming concept in DNN hyperparame- ter optimization.

2. We have utilized a first order type Uncertain<T> to approximate the distributions over the possible hyperparameter values.

3. Describe the algebra to perform computations over the uncertain hyperparameters.

4. Provide syntax to ask boolean question on the uncertain data type to control false
positive and false negative.

5. Improve the performance of random search for hyperparameter optimization.
