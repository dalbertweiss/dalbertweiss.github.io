---
layout: post
comments: true
title: "A Survey on Deep Active Learning"
subtitle: "a review of active learning - a framework that has been heavily used in machine learning when the labelled instance are scarce or cumbersome to obtain"
date: 2020-01-27 23:45:13 -0400
background: '/img/posts/01.jpg'
---

Active Learning is well-motivated in a wide field of real-world scenarios where labeled data are scarce or costly to
acquire. Active learning is a framework in machine learning allowing to reduce the effort for data acquision by making
the assumption that the performance of the machine learning algorithm can be improved when the algorithm is allowed
to choose the data which it wants to learn from. For supervised learning, this can be extended by the idea of being
provided only with a subsample of labeled data instances. The learner sends a quest to the annotator who is supposed
to provide a label for an unlabeled data instance. Afterwards the model is retrained based on the augmented training
instances. This procedure is repeated until the labeling budget is exhausted. 

While active learning has established as a well-known technique when labeled instances are sparse,
it has reached a new paradigm for the application to deep learning. For deep learning settings, the process of active learning has a
contradictory twist. Active Learning is formulated on the basis of querying instances at a time while deep learning
samples from batches during training. Furthermore, while the unlabeled data is used for acquisition the labeled data
is used for training. This is counterintuitive for deep learning models, for which the training of all available data has
proven in better results. Hence, the application of active learning to deep learning portrays a new paradigm, for which a
rephrasing of active learning is inevitable.

While classical active learning selects instances based on uncertainty, the understanding of uncertainty for neural
networks is not straightforward. Uncertainty-based methods sample instances close to the decision boundary. However,
this becomes intractable in case of neural network architectures. Further, it has shown that conventional heuristics such
as confidence show to be misleading [2]. For this reason, recent progress has put strong emphasis on the question of
how to acquire samples for batch aware scenarios.

Within the course of this survey, we aim to shed light behind current approaches applied to deep active learning by
giving a comprehensive summary. Specifically, it should emphasize the need for a reformulation of the active learning
pipeline when applied to deep neural networks. This specifically seek in giving insight to current methods adapted for
batch aware settings. Further, we provide guidelines for future reasearch by identifying prevailing research gaps and
giving further suggestions on state-of-the-art research.

# Problem Formulation
Conventional acquisition functions have proven to not be efficient, resulting in random acquisition to outperform
classical heuristics during model training in some cases [12, 19, 18, 35]. This has motivated the adaption of active
learning for batch-lie settings. However, while strong research has been in the field of finding the appropriate acquisition
strategy, few works have placed attention to the whole process, having lead to deep learning and active learning to
be treated distinctly in many settings [?]. We argue, that a careful integration is crucial in adapting active learning
to deep learning settings such that a reframing of active learning is a prerequisite to simultaneously learn a feature
representation while labeled instances are cumbersome to obtain. These inconsistencies within the active learning
pipeline can be subdivided into the following obstacles:

1. **Insufficiency in the training of the neural network** While classical active learning was defined for training
based on small data, deep learning preserve a data hungry baseline [7]. Thus, while active learing in the
classical sense selects one instance at a time for each training cycle, deep active learning has been adapted
for collecting batches for each training cycle [15]. Furthermore, while labeled instances have been used
for training, unlabeled data serve for acquisition. To enhance training capabilities, learning strategies such
as semi-supervised learning, unsupervised feature learning [17, ?] and data augmentation [41] yield novel
possibilites for an adequate training within the deep active learning pipeline.

2. **Interplay of the active learning pipeline** Current research, as also argued by [20], lack of inconsistencies
within the active learning pipeline. A majority of the proposed active learning strategies have focused on
training classifier based on a fixed feature representation. However, in deep learning the training of the feature
representation is volatile, where the training of both features and classifier are followed in a joint manner. Thus,
to avoid divergence, a integration of deep learning models within the active learning framework is inevitable
[14, 18].

3. **Querying based on uncertainty** For the applicability of active learning to deep neural networks, one has to
weight the cost of acquisition in comparison to the cost of labeling. Query strategies in classical AL framework
concentrate mainly on uncertainty. Although these provide a cheap way of acquisition, they only consider
exploitation, such that the samples within a query batch are sampled close to the decision boundary and, thus,
do not represent the true data distribution. To address this problem, exploration-aware diversity sampling
present a possibility to sample for representative batches of instances. These provide a possibility for the
sampled batches to represent a representative surrogate of the enire data. However, while these are coupled
with higher computational complexity these might provide a waste of resources in the sense of querying for
instances giving without additional information [13].

4. **Explainability and interpretability** The explainablity of deep active learning scenarios compose both the
model interpretability as well as the need for explanation of the acquisition of the sampled batches. While
the former has deserved valuable investigation, there has been no studies that incoorperate it within an active
learning framework. Especially the later deserves investigation for batch sampling to obtain information of the
representativenss of the queried batches, specifically concerning the reasoning of the samples being queried
[21].



# Classical Active Learning
## General Framework
Acquiring labeled data turns out to be costly especially when deep learning is applied. Active Learning is a framework
allowing for a reduced data acquisition when samples are scarce by actively querying for unlabeled data instances,
traditionally approached by sampling based on instances it is most uncertain about. Based on unlabeled data instances,
the learning algorithm may pose queries to an oracle which is usually a human operating as annotator [12]. This is
usually done by choosing a uncertainty sampling strategy, where data instances located close to the decision boundary
are more likey being queried. Lets assume a set of data instances 
<img src="https://render.githubusercontent.com/render/math?math=\mathcal{X} = (x_1, ..., x_N)"> for which 
<img src="https://render.githubusercontent.com/render/math?math=x_i"> represents its feature
vector. Further, lets assume $`Y = (y_1, ..., y_N)$$ being the corresponding set of labels, for which 
<img src="https://render.githubusercontent.com/render/math?math=y_i"> represents the label
for each data instance. As long as the stopping criteria specified by thresh is not reached, the parameter weights <img src="https://render.githubusercontent.com/render/math?math=\theta">
are trained based on the labeled training set L. For this 
<img src="https://render.githubusercontent.com/render/math?math=x"> portrays the most informative data instance chosen by the
sampling strategy φ(x) which is contained within the unnotated data <img src="https://render.githubusercontent.com/render/math?math=\mathcal{U}">
. After annotation this sample is augmented into
X . This procedure is repeated until the labeling budget is exhausted [23, ?].


For active learning, one distinguishes between three different scenarios. Membership query synthesis allows the
learning algorithm to query any unlabeled data instance in the input space, including the instances provided by the
learning algorithm [20]. In stream-based selective sampling a stream of data is passed for annotation which makes the
assumption that querying for an unnotated data instance is inexpensive. In a pool-based scenario a data instances are
derived from a unlabled pool. Although the pool-based scenario is most common within the literature, the stream-based
scenario entail most potential when applied to embedded systems or mobile devices [24, 12, 13].


## Deep Active Learning

Unlike normal data-driven approaches, the selection of the acquisition function has remained hand-crafted in deep active
learning settings. A main concern of applying active learning to deep learning settings remains the lacks the scalibility to
high dimensional data [33, 10]. This can be argued with the fact that deep learning models are trained by a batch-based
manner, while classical active learning is formulated based on sequential querying strategy. Furthermore, to reduce the
cost of annotation the selection of the most representative and informative samples is required. Although there has not
been a clear definition regarding an informative, represenative sampling, informativeness has been assosciated with uncertainty while representativeness concentrates on diversity.
The main distinction between classical active learning and deep active learning relates to the sampling of batches instead
of instances during each iteration. While the training of deep networks based on single instances is ineffective, it may
also favor overfitting. Thus, unlike classical active learning explained in section 4.1, the sampling is executed in batches
such that <img src="https://render.githubusercontent.com/render/math?B={x^*_1,x^*_2,...,x^*_b"> for 
which <img src="https://render.githubusercontent.com/render/math?math=b<N">. Thus, to find the parameterized classifier <img src="https://render.githubusercontent.com/render/math?math=h_\theta"> for a batch-aware setting
it can be written

$$
\begin{aligned}
h_\theta = argmax f
\end{aligned}
$$

To solve this optimization tasks, a valid distinction in choosing the queried batches B relate to their uncertainy and
diversity. This is largely adapted from the classical methods where the difference becomes evident based on the
exploration versus exploitation close to the decision boundary. While samples in batches display a large similarity
within uncertainty methods, diversity tries to capture the full distribution as a surrogate.


However, for deep neural networks a clear understanding of uncertainty is not straightforward. Methods used in the
classical sense such as optimal experiment design [34] would be intractable for Convolutional Neural Networks since it
requires the computation of the inverse of the Hessian matrix during each training iteration [35]. Another issue is the
increasing time complexity required of some algorithms with increasing dimensionality. Thus, to resemble uncertainty
in the sense of neural networks, the softmax response (SR) is used to approximate the activation of an instance.


### Choice of batch-aware acquisition function
The design of the appropriate acquisition function is decisive in terms of the minimization of the labeling cost, playing
a crucial role in the success of the active learning framework. Hence, it is of no surprise that strong emphasis relate to
finding the suiting query strategy. For this we distinguish broadly between uncertainty-based, diversity-based and a
hybrid approach. Although the decision boundary is not tractable for deep networks, we refer to figure 2 which should
give a better understanding between those two approaches. Besides these, we also mention other approaches that do not
inclose within this division.

**Uncertainty-based Methods.** Uncertainty-based methods represent the most studied query strategies due to its
computational simplicity. Although these has been popular for traditional machine learning models, it’s not easy to
apply to deep learning since the understanding of uncertainty is not intuitive [35].
Numerous studies have measured the uncertainty of neural networks according to the softmax response based on the
models outputs. [18] approach this by adapting this to multi-view uncertainty by applying an additional softmax layer
on top of each hidden layer. Afterwards conventional heuristics such as entropy [24], least confidence [37] and margin
sampling [38] can be applied. Fusion of the overall model uncertainty is obtained by calculation of an adaptive weighted
combination. Even though these show to be efficient in case of computational effort, they do not always show good
performance.
Instead Bayesian Neural Networks have become popular technique by providing a distribution of their output parameters.
To allow for inference, an integration of all possible parameters is executed, leading to an ensemble of networks
contributing to the output [10, 41]. For this to be tractable, Monte Carlo dropout is used where dropout is applied
multiple times to approximate the model posterior <img src="https://render.githubusercontent.com/render/math?math=p(\omega|\mathcal{L}"> based on the summation of the outputs. For this dropout is
directly applied at the predicting output. Although the real parameter distribution is unknown, based on the assumption
that they belong to a Bernoulli distribution, the Monte Carlo integration allow for an approximation of the posterior.
Acoordingly, for <img src="https://render.githubusercontent.com/render/math?math=T"> trained neural networks we can approximate

$$
\begin{aligned}
p(y=c|x,L)
\end{aligned}
$$

