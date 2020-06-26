---
layout: post
title: The Layman’s Gaussian Naïve Bayes Classifier
subtitle: What are the odds?
gh-repo: daattali/beautiful-jekyll
gh-badge: [star, fork, follow]
tags: [machine, learning, artificial, network, model, bayes, thorem, gauss]
# bigimg: /img/cdn_bigimg_banner.jpg
comments: true
---
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<img id="cdn_census_area" src="/img/gaussian_header.png" alt="graph">{:height="200%" width="200%"}

## Introduction

The naïve Bayes’ classifiers are a network of simple yet powerful supervised learning algorithms that allows for the discrimination of objects based on features. It is founded on two core principles.

* Assumption of condition independence.
* Bayes’ Theorem.

Although these classifiers work well, they do have some limitations. Having a high bias and a low variance means that it becomes difficult to handle more complex problems and categories.

The naïve Bayes’ classifiers in general, have seen popularity within both academia and real world implementations such as document filtering. Although simple, they have set the groundwork in terms of concepts for more intricate models such as regression.

One of its interpretation is the Gaussian algorithm – a naïve model based on Bayes’ Theorem capable of handling continuous data. This adds an additional principle – continuous values within its classes are distributed according to a normal (or Gaussian) distribution.

## Bayes’ Theorem

$$ P(A\midB) = \frac{P(B\midA) \, P(A)}{P(B)} $$
//
$$P(A\midB) = \frac{P(B\midA) \, P(A)}{P(B)}$$

## Gaussian Naïve Bayes’ algorithm

$\frac{1}{\sigma\sqrt{2\pi}}e^{\frac{1}{2}\left(\frac{\chi-\mu}{\sigma}\right)^{2}}$

## Manual implementation

    class NaiveBayesClassifier():
    """docstring"""
        def __init__(self):

```bash
class NaiveBayesClassifier():
"""docstring"""
    def __init__(self):
```

## Scikit-Learn implementation
