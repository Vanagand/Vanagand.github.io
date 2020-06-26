---
ext-js: "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"
layout: post
title: The Layman’s Gaussian Naïve Bayes Classifier
subtitle: What are the odds?
gh-repo: daattali/beautiful-jekyll
gh-badge: [star, fork, follow]
tags: [machine, learning, artificial, network, model, bayes, thorem, gauss]
# bigimg: /img/cdn_bigimg_banner.jpg
comments: true
---
![header](/assets/img/gaussian_header.png)
## Introduction

The naïve Bayes’ classifiers are a network of simple yet powerful supervised learning algorithms that allows for the discrimination of objects based on features. It is founded on two core principles.

* Assumption of condition independence.
* Bayes’ Theorem.

Although these classifiers work well, they do have some limitations. Having a high bias and a low variance means that it becomes difficult to handle more complex problems and categories.

The naïve Bayes’ classifiers in general, have seen popularity within both academia and real world implementations such as document filtering. Although simple, they have set the groundwork in terms of concepts for more intricate models such as regression.

One of its interpretation is the Gaussian algorithm – a naïve model based on Bayes’ Theorem capable of handling continuous data. This adds an additional principle – continuous values within its classes are distributed according to a normal (or Gaussian) distribution.

## Bayes’ Theorem

$$P(A\mid B) = \frac{P(B\mid A) \, P(A)}{P(B)}$$

## Gaussian Naïve Bayes’ algorithm

$$\frac{1}{\sigma\sqrt{2\pi}}e^{\frac{1}{2}\left(\frac{\chi-\mu}{\sigma}\right)^{2}}$$

## Manual implementation

Now we can take a look at how a **Gaussian Naïve Bayes** model can be done from the ground up. Although there are many ways to implement a model, in this case we can initiate a more general model class called `NaïveBayesClassifier`.

This allows us to create a child class for the various algorithms available – Gaussian being one of them.

```bash
class NaiveBayesClassifier():
""""""
def __init__(self, data):
    self.data = data

# Input decorator.   
@property
def data(self):
    return self._data
@data.setter # Checks input format, else returns an error.
def data(self, x):
    print(f"Evaluating...\n")
    if x != None and isinstance(x, list):
        self._data = x
    else:
        raise ValueError("Please enter a valid dataset.")

```

I’ve also opted to include different mathematical operations that would be otherwise included in packages such as `math` and `numpy` in order to reduce external imports.

```bash
# Mathematical operations.
@property # ratio of a circle's circumference to its diameter.
def piConstant(self):
    return 3.14159265359
@property # base of a natural logarithm.
def eConstant(self):
    return 2.71828182845
def meanFunction(self, x):
    return sum(x) / float(len(x))
def sqrtFunction(self, x):
    return x**(1 / 2)
def expFunction(self, x):
    return self.eConstant**x
def stdevFunction(self, x):
    average = self.meanFunction(x)
    variance = sum([(x - average)**2 for x in x]) / float(len(x) - 1)
    return self.sqrtFunction(variance)
```

Here starts the bulk of the data manipulation. Most of the functions here work in steps, as we complete a task, we call the result in the following function until we have the desired output. We first start by separating the data per class. In the case below we use a dictionary to set the keys as our classes. For our Gaussian function, we also need to retrieve a set of mean, and standard deviations of the various classes, which is shown in the `summary`

```bash
def classSeparate(self, data):
    """Creates a dictionary of feature class values.
    
    Request: data (internal list)
        user input list.

    Returns: var_classDictionary (dictionary)
        dictionary with class value as keys pair. 
    """
    class_dictionary = {}
    for i in range(len(data)):
        vector = data[i]
        class_value = vector[-1]
        if (class_value not in class_dictionary):
            class_dictionary[class_value] = []
        class_dictionary[class_value].append(vector)
    return class_dictionary
@property
def __classdictionary__(self):
    orderedDictclassSeparate = collections.OrderedDict(sorted(self.classSeparate().items()))
    for label, row in orderedDictclassSeparate.items():
        print("class {}".format(label))
        for array in row:
            print("{}".format(array))


def dataSummary(self, data):
    """Creates a list summary of the mean, standard deviation, and lenght of 'n' features.
    
    Request: data (internal list)
        user input list.

    Returns: var_dataSummary (list)
        [
            self.meanFunction(),
            self.stdevFunction(),
            self.len()
        ]
    """
    summary = [(self.meanFunction(x), self.stdevFunction(x), len(x)) for x in zip(*data)]
    del(summary[-1])
    return summary
@property
def __datasummary__(self):
    for row in self.dataSummary(self._data):
        print(row)


def classSummary(self, data):
    """Creates a dictionary summary of the mean, standard deviation, and lenght of 'n' features.
    
    Support function.

    Returns: var_classSummary (dictionary)
        dictionary summary with class value as keys pair. 
    """
    class_dictionary = self.classSeparate(data)
    summary = {}
    for class_value, class_feature in class_dictionary.items():
        summary[class_value] = self.dataSummary(class_feature)
    return summary
@property
def __classsummary__(self):
    orderedDictclassSummary = collections.OrderedDict(sorted(classSummary.items()))
    for label, row in orderedDictclassSummary.items():
        print("class {}".format(label))
        for array in row:
            print("{}".format(array))
```

Now that we have all the information we need to start calculating probabilities, we define a function that takes the summary of a class, and a row, or vector that we can compare against. This function allows us to see what are the chances that an observation is part of a class.

```bash
def classProbability(self, summary, vector):
    """Creates a dictionary of probabilities for each class..
    
    Request: vector (list)
        observation.

    Returns: var_classProbability (dictionary)
        dictionary or probabilities for each class.
    """
    numRows = sum([summary[label][0][2] for label in summary])
    probability = {}
    for class_value, class_summary in summary.items():
        probability[class_value] = summary[class_value][0][2]/float(numRows)
        for i in range(len(class_summary)):
            mean, stdev, _ = class_summary[i]
            probability[class_value] *= self.GaussianProbability(vector[i], mean, stdev)
    return probability
```

Below are general functions, the first which fits the information to our data, and the later that assigns a class to an observation based on its probability. We can loop through this function to get the class probability of multiple observations!

```
def _model(self, train, test):
    summary = self.classSummary(train)
    prediction = []
    for row in test:
        output = self._predict(summary, row)
        prediction.append(output)
    return(prediction)


def _predict(self, summary, vector):
    probability = self.classProbability(summary, vector)
    best_label, best_probability = None, -1
    for class_value, class_probability in probability.items():
        if best_label is None or class_probability > best_probability:
            best_probability = class_probability
            best_label = class_value
    return best_label
```


```bash
class GaussianNBmodel(NaiveBayesClassifier):
def __init__(self, data):
    self.data = data
    self._model_info = "A naive model based on Bayes’ Theorem capable of handling continuous data." \
    "It is assumed that the continuous values within their classes are distributed according to a normal " \
    "(or Gaussian) distribution."
@property
def about(self):
    return self._model_info
```

In the scenario that we want to implement more algorithms, we can create child classes that retains the information and equation of their respective algorithms. This way we can create an object whose purpose is to provide predictions based on a single algorithm. The equation below is known as the **Gaussian distribution function**.

```bash
def GaussianProbability(self, data, mean, stdev):
    """Creates a gaussian probability distribution float.
    
    Request: data (internal list)
        user input list.
    mean: float
    stdev: float
        [
            self.data,
            self.meanFunction(),
            self.stdevFunction()
        ]

    Returns: float
        float value of the gaussian probability of events.
    """
    exponent = self.expFunction(-((data - mean)**2 / (2 * stdev**2)))
    return (1 / (self.sqrtFunction(2 * self.piConstant) * stdev)) * exponent
@property
def __gaussianprobability__(self):
    print("{:.3f}".format)
```

Now we can check the general accuracy of the model!
We can tie a bow here by making a prediction on an arbitrary set of features!  Ill split the dataset and will explain more below while we test our results against the Scikit-Lean implementation. 

```bash
# Data import
data = load_iris()
df_iris = pd.DataFrame(data)

# Dataset split
X_feature = df_iris.iloc[:,0:4]
y_class = df_iris.iloc[:,4]

# Fitting the model
model._model(dataIrisFeature, dataIrisTarget)
# Printing the output of an arbitrary class prediction.
print(model._predict(modelFit, [5.1,3.5,1.4,0.2]))
>>> [0]
```

As we can see, our observation would belong to the class `[0]`.

## Scikit-Learn implementation

We can also test our model against the Scikit-Learn package. Which also contains the popular iris dataset for classification. We can start by converting our raw dataset into a `Dataframe` object for further manipulation.  

```bash
import pandas as pd
from sklearn.datasets import load_iris

data = load_iris()
df_iris = pd.DataFrame(data)
```

We can also split our entire dataset into a feature subset, which contains the relevant information in order to deduce our second subset of data – the class. Depending on how we manipulate the original data, we might need to encode the class features, such as **setosa**, **versicolor**, and **virginica** to integer values. 

```bash
X_feature = df_iris.iloc[:,0:4]
y_class = df_iris.iloc[:,4]
```

The last step is to fit our model on the training data. Here the model finds the coefficients for the equation specified via the algorithm being used. Then, we could classify incoming data using a prediction method. 

In our case we use the **Gaussian Naïve Bayes** from the Scikit-Learn package for our calculations. We can then predict an arbitrary row to see what class the model would give the observation.


```bash
from sklearn.naive_bayes import GaussianNB

gaussianClassifier  = GaussianNB()
gaussianClassifier.fit(X_feature, y_class)

print(gaussianClassifier.predict([[5.1,3.5,1.4,0.2]]))
>>> [0]
```

After this, we can check the general accuracy of the model! 

```bash
print(gaussianClassifier.score(X_feature, y_class))
>>> 96%
```

