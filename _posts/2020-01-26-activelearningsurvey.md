---
layout: post
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

1. Insufficiency in the training of the neural network While classical active learning was defined for training
based on small data, deep learning preserve a data hungry baseline [7]. Thus, while active learing in the
classical sense selects one instance at a time for each training cycle, deep active learning has been adapted
for collecting batches for each training cycle [15]. Furthermore, while labeled instances have been used
for training, unlabeled data serve for acquisition. To enhance training capabilities, learning strategies such
as semi-supervised learning, unsupervised feature learning [17, ?] and data augmentation [41] yield novel
possibilites for an adequate training within the deep active learning pipeline.

2. Interplay of the active learning pipeline Current research, as also argued by [20], lack of inconsistencies
within the active learning pipeline. A majority of the proposed active learning strategies have focused on
training classifier based on a fixed feature representation. However, in deep learning the training of the feature
representation is volatile, where the training of both features and classifier are followed in a joint manner. Thus,
to avoid divergence, a integration of deep learning models within the active learning framework is inevitable
[14, 18].

3. Querying based on uncertainty For the applicability of active learning to deep neural networks, one has to
weight the cost of acquisition in comparison to the cost of labeling. Query strategies in classical AL framework
concentrate mainly on uncertainty. Although these provide a cheap way of acquisition, they only consider
exploitation, such that the samples within a query batch are sampled close to the decision boundary and, thus,
do not represent the true data distribution. To address this problem, exploration-aware diversity sampling
present a possibility to sample for representative batches of instances. These provide a possibility for the
sampled batches to represent a representative surrogate of the enire data. However, while these are coupled
with higher computational complexity these might provide a waste of resources in the sense of querying for
instances giving without additional information [13].

4. Explainability and interpretability The explainablity of deep active learning scenarios compose both the
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
are more likey being queried. Lets assume a set of data instances $`X = (x_1, ..., x_N)`$ for which $`x_i`$ represents its feature
vector. Further, lets assume $`Y = (y_1, ..., y_N)`$ being the corresponding set of labels, for which $`y_i`$ represents the label
for each data instance. As long as the stopping criteria specified by thresh is not reached, the parameter weights θ
are trained based on the labeled training set L. For this $x$ portrays the most informative data instance chosen by the
sampling strategy φ(x) which is contained within the unnotated data U. After annotation this sample is augmented into
X . This procedure is repeated until the labeling budget is exhausted [23, ?].