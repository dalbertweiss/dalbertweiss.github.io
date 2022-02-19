---
layout: post
title: "Continual Learning for learning sequentially"
subtitle: "lifelong learning when tasks enter sequentially within the data science pipeline"
date: 2022-01-17 23:45:13 -0400
background: '/img/posts/01.jpg'
---

Continual learning is a new branch within deep learning which tries to infer on new tasks based on previously aquired knowledge where data becomes incrementally available <a href="https://www.sciencedirect.com/science/article/pii/S0893608019300231">[1]</a>. While in deep learning the assumption is usually made processed data come from the same data distribution, this assumption can be broken in *continual learning settings*. Here, the algorithm can adapt to novel information with reaching the situation of forgetting previously acquired knowledge. This so called *learning without forgetting* is an active research usually accomplished by replaying old memory within the new learning task. 


# 1. Introduction to Continual Learning
Within deep learning it is often assumed that all data is simultaneously available where it is assumed that training and testset come from the same distribution. While sequential arriving data has often been tackled by *online learning* has often been referred to when data points enter sequentially, continual learning offers new opportunities when data arrives for *a new task*. This differs in such a way that the assumption is made that the data can be subdevided into *tasks* allowing to break the iid assumption. A task refers to new training data characterized by a new class, a new domain or different output space <a href="https://ieeexplore.ieee.org/document/9349197">[2]</a>. This setting is called *disjoint task setting* and for classification purposes often called *class-incremental setting*.

> **Definition (task):** A tasks is an abstract representation for a learning experience characterized by a unique task label <img src="https://render.githubusercontent.com/render/math?math=t"> where the target funciont is given by the objective.

While continual learning has its fundamentals in robotics, the approaches are often been inspired by biochemical interaction within the brain. For example, the idea of using replay methods to replay previous information inside the memory has been inspired by the processing of data in the hyppocampus. The idea is to tackle the *problem of forgetting* which often is encountered when facing different tasks. In this situation the neural network will adapt to the last learned task while forgetting the previously learned tasks (plasticity-stability dilemma). Due to the shift of the data distribution a phenomenon often referred to as *concept drift* can take place. 

> **Definition (catastrophic forgetting):**  Given a sequence of n supervised learning tasks <img src="https://render.githubusercontent.com/render/math?math=T=(T_1,...,T_n)">, the aim is to learn a system that performs well on every new task while not forgetting previously acquired knowledge.

Generally, one distinguishes between the following continual learning scenarios:

- **Task-Incremental Learing**: For a set of tasks the task identity is known.
- **Domain-Incremental Learning**: For a set of tasks, the task identity is not provided during testing and does not be inferred.
- **Class-Incremental Learning**: For a set of tasks the task identity is not provided during testing and needs to be inferred.


# 2. Biological Interplay with Lifelong learning
## a. Stability-plasticity dilemma
Molecular gradients and interactions lead to advances in the regard of developing new tasks and transfer knowledge across domains. Although it might happen that tasks and trained tasks can be forgotten within a lifespan this rarely is caused by catastrophically forgetting previously acquired skills. Lifelong learning, also referrred to as continual learning, has drawn many of its inspirations from biological interactions and interplay within the brain. Within this context the stability-plasticity dilemma is often mentioned to judge how the learning of new tasks relates with the forgetting of old tasks.

Neurosynaptic plasticity is a feature of the brain, allowing to lean, modify and adapt to dynamic and reevolving environments. It has been shown that the plasticity of the brain particularly becomes available when difficult situations. This is specifically the case in post-developmental situations shown to be correlated with a decreasing levels of plasticity [1]. Further research provide evidence that within the mammalian (mouse) brian the excitatory synapses are strengthend when learning novel tasks while further evidence has shown an increase in the volume of dendritic spines of neurons [Yang et al, 2009,Overcoming catastrophic forgetting in neural networks].






## b. Hebbian Plasticity and Stability
The interaction and connectivity within the cortex, very much influences how information is processed and how learning takes place. The first documented interaction of reaching synaptic plasticity  for how external stimuli provides a way of driving neural activity was proposed by Hebb (1949). Supposing Hebb's rule with states how the interplay of postsynaptic cells with presynaptic cells can lead to the devoplment of neural connectivity. Suppose of assuming a learning rate <img src="https://render.githubusercontent.com/render/math?math=\eta"> and a synaptic strength  <img src="https://render.githubusercontent.com/render/math?math=w"> which is updated by the prodcut of the pre- and post-synatpic activy  <img src="https://render.githubusercontent.com/render/math?math=x"> and  <img src="https://render.githubusercontent.com/render/math?math=y">

$$
\begin{aligned}
\Delta w = x \cdot y \cdot \eta
\end{aligned}
$$




# 3. Types of Settings for Continual Learning Approaches
Problems encountered during continual learning might be the shifts of data distributions, unbalanced data or the problem of catastrophic forgetting. While there has been requirements set upon continual learning learning, often referred to as the *desirata of continual learning*, these mainly describe the rules which should have been satisfied for which we refer to resources within [3]. 


> **Definition (continual learning):**  Assuming an infinite number of arriving sequentially data streams, continual learning setting allows to transfer previously previously learned task knowledge to novel applications, with the desire to not forget previously acquired information. For this, different tasks do not need to obey the assumption that the data come from the same data distribution
may be violated. 

Given that <img src="https://render.githubusercontent.com/render/math?math=T"> defines the tasks faced during training, the empirical risk may be described by

$$
\begin{aligned}
R = \frac{1}{N_t} \sum_{t=1}^T \sum^N_{n=1} l(f(x_n^t;\theta), y_n^t)
\end{aligned}
$$

where <img src="https://render.githubusercontent.com/render/math?math=N"> recalls the available samples [3].

To encounter this problem different kind of methods have established often subdevided into *replay*, *regularization-based* and *parameter isolation* methods. Regularization-based approaches have been used control the variance without really having an effect on the bias. Replay methods have been used to either replay old data or considering data consisting of the same data distribution
as the older learned tasks. Parameter isolation methods refer to methods that freeze parts of the neural network __. While for the first case, these are usually based on randomly shuffling previously memory, there have been novel approach to sample data that is most representitive. For a good overview of the different methods shown in Fig. 1.

![image-in-text](/img/posts/continual-learning-1.PNG)
*Fig. 1. Illustration of various famous continual learning methods [4].*




### a. Regularization-based methods
Regularization based methods allow
These simple regularization techniques reduce the chance of weights being
modified, and thus decrease the probability of forgetting.

**Elastic Weight Consolidation** relies on Bayesian learning for which the posterior of the previous task contributes to the prior of the new task. Since the computation of the posterior in deep learning is intractable, an estimate based on the Laplacian approximation is determined. The precision herefore is estimated using the Fisher Information Matrix. 
For better understanding assume that for every parameters optimized during training of a deep learning model, such that <img src="https://render.githubusercontent.com/render/math?math=\theta = \theta^*">. After learning a training a task <img src="https://render.githubusercontent.com/render/math?math=\mathcal{A}">, the regularization method penalizes task <img src="https://render.githubusercontent.com/render/math?math=\mathcal{A}"> for learning the parameters of task <img src="https://render.githubusercontent.com/render/math?math=\mathcal{B}">. To calculate the posterior distribution, we can state that

$$
\begin{aligned}
log(p(\theta|\sum)) = log(p(\sum|\theta)) + log(p(\theta)) - log(p(\sum))
\end{aligned}
$$

However, since the calcuation of the posterior becomes intractable, there is no direct calculation of the quantiles. To bypass this, the Laplacian approximation is used, applying a fitting based on the use of the normal distribution to approximate the probability density function.

![Laplacian Approximation](/img/posts/continual-learning-2.png)


**Incremental Moment Matching (IMM)**

**Synaptic Intellgence**


**Rotation-EWC**
**MAS**
**Riemannian Walk**
**Learning without forgetting (LwF)**


### b. Replay methods
Replay methods stores previous raw data or generates pseudo-samples using generative models, replayed within learning the new task to alleviate forgetting. One of the most simple approaches is to store old data within a memory buffer inspired by experience replay often used in reinforcement learning (episodic memory system). Another approach is **iCaRL** stores a subset of examplars per class building the proxy of the class mean. iCaRL makes the assumption of a class-incremental setting where the data arriving sequentially is equipped with novel classes. This approach not only model agnostic but also task agnostic at the same time.


Although using replay methods is an efficient method to avoid forgetting older tasks it does not only require large memory capacity but also is inefficienct in real world scenarios such as the usage of edge devices where the storage of old data has to be provided <a href="https://arxiv.org/abs/1705.08690">[3]</a>. Another approach is to generate pseudo-data using generative models. Continual learning with deep generative replay by using a dual architecture consisting of a generator to generate new data and a solver. This idea is inspired by the dual memeory system in the hippocampus and the neocortex. While the hippocampus encodes this recent experience, the memory is consolidated in neocortex through. The idea of generating novel data using a generative adversial networks (GANs) is used to imitate old data. Using this approach a task solver is used to pair the generated data with label. This allows to generate a tuple of input and target signal without having access to the old training data.

![image info](../img/posts/continual-learning-0.png)

![Autoencoder architecture]({{ '/img/posts/continual-learning-0.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 1. Illustration of autoencoder model architecture.*



### c. Parameter isolation methods



# 3. Application to MNIST dataset


# 4. Conclusion


# 5. 


[1]
[2]
[3] Continual Learning in Neural Networks