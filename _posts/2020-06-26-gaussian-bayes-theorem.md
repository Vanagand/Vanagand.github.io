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



```bash
class NaiveBayesClassifier():
"""docstring"""
    def __init__(self):
```

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
print(gaussianClassifier.score(X_data, y_labels))
```

After this, we can check the general accuracy of the model! 

```bash
print(gaussianClassifier.score(X_feature, y_class))
```