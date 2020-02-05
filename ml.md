# Machine Learning

Machine Learning is making the computer learn from studying data and statistics.

Machine Learning is a step into the direction of artificial intelligence (AI).

Machine Learning is a program that analyses data and learns to predict the outcome.


## Where To Start?
In this tutorial we will go back to mathematics and study statistics, and how to calculate important numbers based on data sets.

We will also learn how to use various Python modules to get the answers we need.

And we will learn how to make functions that are able to predict the outcome based on what we have learned.



## Data Set


## Algo


- Linear Regression

Linear regression uses the relationship between the data-points to draw a straight line through all them.

This line can be used to predict future values.



```py
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

plt.scatter(x, y)
plt.show()
```

```py
from scipy import stats

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()
```

## Other
- Logistic Regression
- k-Nearest Neighbor (kNN)
- Convolutional Neural Network (CNN)
- Recurrent Neural Networks (RNNs)


## Learning paradigms


-  Supervised learning
uses a set of paired inputs and desired outputs. The learning task is to produce the desired output for each input.
eg linear regression

- Unsupervised learning
type of learning used to draw inferences from datasets consisting of input data without labeled responses.
eg cluster analysis

- Reinforcement learning
In applications such as playing video games, an actor takes a string of actions, receiving a generally unpredictable response from the environment after each one.

## Deep Learning

## Intro
a subfield of machine learning concerned with algorithms inspired by the structure and function of the brain called artificial neural networks.


Two of the top numerical platforms in Python that provide the basis for Deep Learning research and development are `Theano` and `TensorFlow`.

## Neural Nets aka (Artificial neural networks (ANN) or connectionist systems )

Neural networks are a set of algorithms, modeled loosely after the human brain, that are designed to recognize patterns. They interpret sensory data through a kind of machine perception, labeling or clustering raw input.
