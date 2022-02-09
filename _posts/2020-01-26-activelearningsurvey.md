---
layout: post
title: "A Survey on Deep Active Learning"
subtitle: "a review of active learning - a framework that has been heavily used in machine learning when the labelled instance are scarce or cumbersome to obtain"
date: 2020-01-27 23:45:13 -0400
background: '/img/posts/01.jpg'
---



<p>Active Learning is well-motivated in a wide field of real-world scenarios where labeled data are scarce or costly to
acquire. Active learning is a framework in machine learning allowing to reduce the effort for data acquision by making
the assumption that the performance of the machine learning algorithm can be improved when the algorithm is allowed
to choose the data which it wants to learn from. For supervised learning, this can be extended by the idea of being
provided only with a subsample of labeled data instances. The learner sends a quest to the annotator who is supposed
to provide a label for an unlabeled data instance. Afterwards the model is retrained based on the augmented training
instances. This procedure is repeated until the labeling budget is exhausted. </p>

<p>While active learning has established as a well-known technique when labeled instances are sparse,
it has reached a new paradigm for the application to deep learning. For deep learning settings, the process of active learning has a
contradictory twist. Active Learning is formulated on the basis of querying instances at a time while deep learning
samples from batches during training. Furthermore, while the unlabeled data is used for acquisition the labeled data
is used for training. This is counterintuitive for deep learning models, for which the training of all available data has
proven in better results. Hence, the application of active learning to deep learning portrays a new paradigm, for which a
rephrasing of active learning is inevitable. </p>

<p>While classical active learning selects instances based on uncertainty, the understanding of uncertainty for neural
networks is not straightforward. Uncertainty-based methods sample instances close to the decision boundary. However,
this becomes intractable in case of neural network architectures. Further, it has shown that conventional heuristics such
as confidence show to be misleading [?, 2]. For this reason, recent progress has put strong emphasis on the question of
how to acquire samples for batch aware scenarios.</p>

**test test**

<h2 class="section-heading">The Active Learning Framework</h2>

Acquiring labeled data turns out to be costly especially when deep learning is applied. Active Learning is a framework
allowing for a reduced data acquisition when samples are scarce by actively querying for unlabeled data instances,
traditionally approached by sampling based on instances it is most uncertain about.
For active learning, one distinguishes between three different scenarios. Membership query synthesis allows the learning
algorithm to query any unlabeled data instance in the input space, including the instances provided by the learning
algorithm [3]. In stream-based selective sampling a stream of data is passed for annotation which makes the assumption
that querying for an unnotated data instance is inexpensive. In a pool-based scenario a data instances are derived from a
unlabled pool. Although the pool-based scenario is most common within in the literature, the stream-based scenario
entail most potential when applied to embedded systems or mobile devices.
Based on unlabeled data instances, the learning algorithm may pose queries to an oracle which is usually a human
operating as annotator. This is usually done by choosing a uncertainty sampling strategy, where data instances which
are located close to the decision boundary are more likey being queried. For a pool-based scenario, lets assume a set of
data instances X = (x1, ..., xN ) for which xi represents its feature vector. Further, lets assume Y = (y1, ..., yN ) being
the corresponding set of labels, for which yi represents the label for each data instance. As long as the stopping criteria
specified by thresh is not reached, the parameter weights θ are trained based on the labeled training set L. For x
∗
portraying the most informative data instance for some chosen sampling strategy φ(x), x represents the data instance in
an pool of unnotated data U. This procedure is repeated until the labeling budget is exhausted.

<img class="img-fluid" src="https://source.unsplash.com/Mn9Fa_wQH-M/800x450" alt="Demo Image">
<span class="caption text-muted">To go places and do things that have never been done before – that’s what living is all about.</span>

<p>Space, the final frontier. These are the voyages of the Starship Enterprise. Its five-year mission: to explore strange new worlds, to seek out new life and new civilizations, to boldly go where no man has gone before.</p>

<p>As I stand out here in the wonders of the unknown at Hadley, I sort of realize there’s a fundamental truth to our nature, Man must explore, and this is exploration at its greatest.</p>

<p>Placeholder text by <a href="http://spaceipsum.com/">Space Ipsum</a>. Photographs by <a href="https://unsplash.com/">Unsplash</a>.</p>