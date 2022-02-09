---
layout: post
title: "Continual Learning for learning sequentially"
subtitle: "lifelong learning when tasks enter sequentially within the data science pipeline"
date: 2020-01-27 23:45:13 -0400
background: '/img/posts/01.jpg'
---

# 1. Introduction to Continual Learning
Within deep learning it is often assumed that all data is simultaneously available where it is assumed that training and testset 
come from the same distribution. While sequential arriving data has often been tackled by *online learning* has often been referred to when data points enter sequentially, continual learning offers new opportunities when data arrives for *a new task*. This differs in such a way that the assumption is made that the data can be subdevided into *tasks* allowing to break the iid assumption. This setting is called *disjoint task setting* and for classification purposes often called *class-incremental setting*.

While continual learning has its fundamentals in robotics, the approaches are often braininspired. For example, the idea of using replay methods to replay previous information inside the memory has been inspired by the processing of data in the hypocampus. The idea is to tackle the *problem of forgetting* which often is encountered when facing new tasks.



# 2. Types of Settings for Continual Learning Approaches
Replay methods
Regularization-based methods
Parameter isolation methods


## a. Regularization-based methods


## b. Replay methods


## Paramter isolation methods

# 3. Application to MNIST dataset


# 4. Conclusion