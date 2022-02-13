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
vector. Further, lets assume <img src="https://render.githubusercontent.com/render/math?math=Y=(y_1,...,y_N)"> being the corresponding set of labels, for which 
<img src="https://render.githubusercontent.com/render/math?math=y_i"> represents the label
for each data instance. As long as the stopping criteria specified by thresh is not reached, the parameter weights <img src="https://render.githubusercontent.com/render/math?math=\theta">
are trained based on the labeled training set <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}">. For this 
<img src="https://render.githubusercontent.com/render/math?math=x"> portrays the most informative data instance chosen by the
sampling strategy φ(x) which is contained within the unnotated data <img src="https://render.githubusercontent.com/render/math?math=\mathcal{U}">
. After annotation this sample is augmented into <img src="https://render.githubusercontent.com/render/math?math=\mathcal{U}">. This procedure is repeated until the labeling budget is exhausted [23, ?].


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

![image-in-text](/img/posts/activelearning-0.PNG)

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
multiple times to approximate the model posterior <img src="https://render.githubusercontent.com/render/math?math=p(\omega|\mathcal{L})"> based on the summation of the outputs. For this dropout is
directly applied at the predicting output. Although the real parameter distribution is unknown, based on the assumption
that they belong to a Bernoulli distribution, the Monte Carlo integration allow for an approximation of the posterior.
Acoordingly, for <img src="https://render.githubusercontent.com/render/math?math=T"> trained neural networks we can approximate

$$
\begin{aligned}
p(y=c|x,L) = \sum p(y=c|x,\omega)p(\omega |L) d\omega = \frac{1}{T} \sum_{i=1}^T p(y=c|x,\hat{\omega}_t)
\end{aligned}
$$

where <img src="https://render.githubusercontent.com/render/math?math=c"> is the class category and <img src="https://render.githubusercontent.com/render/math?math=\omega"> represent the model parameters, respectively. In Baysian Active Learning by
Disagreement (BALD) the goal is to maximize the mutual information between the prediction and model posterior:

$$
\begin{aligned}
\mathbb{I}[y,\omega | \mathbf{x}, \mathcal{L}] = \mathbb{H[y|\mathbf{x},\mathcal{L}]}-
\mathbb{E}_{p(\omega|\mathcal{L})}[\mathbb{H}(y|\mathbf{x},\omega)] 
\end{aligned}
$$

Maximizing the mutal information implies that the entropy <img src="https://render.githubusercontent.com/render/math?math=\mathbb{H}[y | \mathbf{x},\mathcal{L}]"> has to be of high value corresponding to a high uncertainty of the models prediction. Further, the expected value of entropy is low when it is certain based on the model parameters drawn from the posterior. BatchBALD provides an extension of BALD by acquiring batches of instances at once, thus, leading to a reformulation of


$$
\begin{aligned}
\mathbb{I}[y_{1:b},\omega | x_{1:b}, \mathcal{L}] =
\mathbb{H}[y_{1:b}|x_{1:b}, \mathcal{L}] - \mathbb{E}_{p(\omega|\mathcal{L})
\mathbb{H}(y_{1:b}| x_{1:b}, \omega )]
\end{aligned}
$$



where <img src="https://render.githubusercontent.com/render/math?math=b"> represents the batch of data points and <img src="https://render.githubusercontent.com/render/math?math=(y_1,...,x_b)"> and <img src="https://render.githubusercontent.com/render/math?math=(x_1,...,x_b)"> are abbreviated by <img src="https://render.githubusercontent.com/render/math?math=y_{1:b}"> and <img src="https://render.githubusercontent.com/render/math?math=x_{1:b}"> due to writing purposes.

While bayesian methods have been studied for uncertainty sampling, a further choice of acquisition has been subjected
to semi-supervised methods. Generative Adversarial Active Learning (GAAL) uses generative models to synthesize
instances based on uncertainty during each AL cycle. For this, within the minimization of the optimization problem the
instance of the pool is replaced by a generator. Then the objective function is optimized based on gradient descent.
While tested for both MNIST and CIFAR-10 dataset, it performs less good than random acquisition which can be
reasoned by sampling bias [?, 12, 13].
A similar approach for generating uncertainty-based samples based on generative models have been proposed based
on generating Adversarial Sampling for Active Learning (ASAL). This GAN-based method generates and retrieves
samples for annotation based on similarity for multi-class classification problems [13]. In comparison to DCGAN bein
used in GAAL, they replace it by a Wasserstein GAN achieving better results. [13] report that reasons for the weaker
performance of GAAL relate to the identification of labels of generated images can be difficult to manage

![image-in-text](/img/posts/activelearning-1.PNG)




**Diversity-based Methods** Diversity-based methods select data points diversily throughout the feature space, compensating the lack of exploration within uncertainty-based methods. This idea is contratry to the idea of uncertainty
sampling and has usually been tackled by geometric approaches. This is approached by taking a set of selected samples,
usually referred to as core set, to represent the distribution of the feature space of the entire training set [11, 12]. For
this usually the batch size is increased to avoid for the sampling bias problem.
The most renowned approach defined active learning as a coreset selection problem where the selection of a subset
of data instances from a model trained to achieve the same performance as the model on the entirely available data
instances. For this the upper bound of the loss function is minimized, proven to be equivalent to the k-center cluster problem visualized in figure 3. However, since this describes a NP-hard problem, a discrete greedy optimazation
algorithm is used to solve it in a P-hard manner [12]. However, this usually required the computation of a large
distance matrix on the unlabeled dataset, leading to high computational complexity [20]. Futher, the coreset approach is
particularly of a bad choice when the unlabeled pool is large with simulatenously having a small query batch [13].
Farthest-First Compression (FF-Comp) proposes a compression scheme that uses farthest-first traversals in the space of
neural activation over a representation layer to query contiguous instances from the pool [11]. Inspired by the coreset
approach, this allows to retrieve a 2-approximation algorithm of the k-center problem.
Other diversity approaches try to incoorperate the unlabeled data directly. For this [15] clusters the unlabeled data pool
U based on k-means algorithm, choosing instances located close to the cluster center.



**Hybrid approach** While uncertainty-based method sample close to the decision boundary, diversity-based methods
allow to query more explorative instances with the cost of being more prone for the selection of outliers. However,
while uncertainty lacks the ability to capture the distribution of the data, diversity metohds might not take might lack
to consider the task itself and, thus, lead to redundant sampling. To compensate these drawbacks, there have been
approaches to combine both of them.
Wasserstein Adversarial Active Learning (WAAL) unites uncertainty and diversity by formulating active learning as
a distribution matching problem [39]. They show that for encapturing diversity, the Wasserstein distance is a better
metric than the H-divergence usually used for measuring diversity of a query batch. For this WAAL in comprises of
two steps, namely a min-max optimization for the DNN parameters and a query batch selection. In comparison to the
diversity-based methods based on core-sets and k-Median [12], WAAL shows a much faster query time, which can be
explained with them requiring the computation of the feature space [39].
Task-Aware Variational Adversarial Active Learning (TA-VAAL) [40] integrating the loss prediction module and the
concept of RankCGAN into Variational Adversarial Active Learning (VAAL). For this the conditional VAE (cVAE) is
joined with a rank variable while the task learner is equiped with a ranker passing the loss rank information to the rank
variable in the cVAE. While considering both labeled and unlabeled data for training this, the convince with a higher
performance when the hybdrid method with standard methods based on uncertainty or diversity [40].
Batch Active learning by Diverse Gradient Embedding (BADGE) incoorparates uncertainty and diversity without tuning
another hyperparameter by sampling groups of points that are disparate and high magnitude in a (hallucinated) gradient
space. This is accompanied by a two step process entailing a gradient embedding and a sampling step. For this model
uncertainty is reflected by the difference in the gradient of the loss. For the assumption is made that when the label of
an instance induces a large gradent of the loss this linked with high model uncertainty. Thus, the gradient embedding is appropriate measure for uncertainty where a minimal norm of the gradient embedding is assosciated with a low
uncertainty. Within the gradient embedding step the label of the instance of the current model is computed folodg by
the calculation of the gradient of the loss with respect to the parameters of the last layer of the network. Afterwards a
sampling step is executed for which a set of points are selected via k-MEANS++ initialisation acting as a workaround
for k-Determinantal Point Process (k-DPP) allowing to select diverse batches of high magnitude without defining an
additional hyperparameter compensating between uncertainty and diversity samples.


## 5.2 Model Training
This section addresses the second point mentioned within section 2. While active learning relies on small amount of
labeled instances, deep learning models are known for being data-hungry. To leverage this problem, strategies were
proposed that allow for compensating reduced training data [24][?][?]. Conventional model training in active learning
solely evolves based on labeled data instances, resulting in unused resources in terms of neglecting the existence of
unlabled data instances. To increase the number of available data during the training process and, thus, train the neural
network architecture to its full extension, methods have been proposed to enhance the training of the model within the
loop.
The Cost-effective active learning (CEAL) strategy does so by including instances of high confidence within the training
by providing them with pseudo-labels. These pseudo-labels are inferred based on the networks prediction and has the
advantage of not being provided with additional labeling costs in terms of queries. However, CEAL is supplied with
another hyperparameter thresholding the prediction’s confidences of the network. Under the circumstance, that the
hyperparameter is falsely adjusted, it can lead to distortions of the label of the training set [14].
Further enlargement of the training set has been proposed by data augmentation using Generative Adversarial Networks.
Generative adversarial active learning (GAAL) generates novel samples with the aim of generating samples with an
addition information gain [36]. However, random data augmentation is not a warranty for an information gain and
could thus be of no additional benefit while wasting computational resources. Furthermore, it is limited by choosing
simple heuristics since these will be used within the generative model. However, it has been shown that these do not
perform well in a deep active learning setting [10, 41]. For this reason [41] introduced Bayesian Generative Active Deep
Learning (BGADL). While [36] train both generator and classifier disjointly, [41] incoorporates the generative and
classification model within one step. For this the idea of GAAL is extended by introducing a variational autoencoder
generative adversarial network (VAE-GAN) in which both Variational Autoencoder (VAE) and GAN are linked by a
VAE decoder. The VAE decoder represents the generator of the GAN model. Opposed to these approaches, DeepFool
Active Learning (DFAL) tries to use a margin-based approach by sampling for their adversarial attacks by taking not
only the unlabeled samples into regard but also its adversarial counterpart [?].
[42] point out that tradition techniques do not scale well to deep neural networks. Further, they argue that geometric
approaches such as the core-set technique suffer of the distance concentration phenomenon and do not scale well when
the number of classes increase. To overcome this issue, they propose Variational Adversarial Active Learning (VAAL)
to learn a lower dimension of the latent represenation on both labeled and unlabeled data based on a semi-supervised
strategy. A variational autoenconder (VAE) is trained to learn the distribution of the labeled data, serving to fool the
discriminator in an adversarial network that the samples belong to the labeled data. Meanwhile the discriminator is
trained to differentiate between labeled and unlabeled data. This method shows to be task agnostic since it does not
depend on the task itself.
Similarly, [?] use conditional Generative Adversarial Networks (cGAN) to generate realistic images used during
training. While both the approaches of [41] and [42] use only the labeled data for training, [43] proposed a strategy to
entail both unlabeled and labeled data within the training process. [43] replaces supervised learning by suggesting an
semi-supervised learning approach by co-training on both labeled and unlabeled data instances. This is approached by
embedding GAN within the classifier to infer the class labels of the generated images. [17] try out a similar approach
bei using both unsupervised and semi-supervised learning to integrate both labeled and unlabeled instances within the
training process. While unsupervised learning is used at the start of the active learning pipeline for model initialization
during each cycle, semi-supervised learning serve for training on all the data simultaneously. As they show within their
finding, this apporach yielded a significant improvement of the performance.
Opposed to the mentioned methods, iNAS has been introduced, arguing while most reasearch has been dedicated to the
design of the appropriate acquisition function, the architecture of the neural network is assumed to be well-suited for
the active learning. To compensate this problem [11] proposed use a large labeled training set for neural architecture optimization and use active learning over the long tail. Instead [44] introduce incremental neural architecture search
(iNAS) which uses a neural architecture search (NAS) for dynamically searching for the optimal architecture on-the-fly
integrated within the active learning pipeline. To avoid overfitting within the initial stages, iNAS starts with a small
capacity incrementally increased with the rise in labeled data. They are able to show that unlike with a preassumed
architecture iNAS allows to perform better than the fixed one [44].

## 5.3 Explainability of deep active models
ecent progresses have tried to enhance the explainability and interpretablity of active learning by developing local
explainers that allow to specify the reason for a certain query. Local Interpretable Model-agnostic Explanations
framework (LIME) has been used in conjunction with active learning to justify the reason of a queried instance. LIME
allows to make faithful explanations by producing pertubations of an instance and interprets the results based on its
hypothetical neighborhood. This allows the interpretation of the uncertainty of each point [45]. However, [21] argue
although this approach is model agnostic, it lacks the ability of the local model being a good approximation of the
classifier. Instead they propose self-explainable neural networks (SENN) which offer explainibility of their prediction.

# 7 Open research questions
ecent successes of deep active learning convey novel topics of research for which most preassumptions made have
been left uncommented. For this reason, we want to point out on open research questions which have to be tackled
within future academia:

**Batch training** Algorithms such as [14] [?] incoorperate training based on batches. Advantages is the
improved training. However, it is still unclear how these batches should be collected within practice.

**Pool scenario** Similar things apply as for the previous point mentioned. Most algorithms are specified for
pool-based scenarios. However, it is not clear how the pool of data should be collected within an active
learning framework.

**Deep Model Architecture** When active learning is applied a model a tuned deep model architecture is assumed
for granted. Usually, the active learning model has been imposed on this previous optimized architecture [15].
[?] propose a reverse approach in which the deep model architecture is learned on-the-fly. Keeping this in
mind, this might pose further difficulties when selecting the appropriate acquisiton function for one’s problem.
Semi-supervised Training While

**Class inbalance** Current works have mainly been tested on freely available datasets showing high class
balance. However, in natural circumstances one is faced with class inbalance which can cause overfitting of
minor classes. For this especially active learning provides in interesting setting which has only been taken into
regard within a few studies. In [?] fairness was

**Budget of annotation** When deciding for the acquisition function to be used in active learning, one has to
take the budget for labeling into regard. This is specifically to mention when comparing uncertainty and
diversity-based methods for querying. One reason for the establishment of more uncertainty-based query
strategies can be related to it being less computational expensive. While this might drive further progress of
their establishment, these do not compensate for the recognition of the distribution of the data.
Informativeness and representativeness While uncertainty-based methods often encounter sampling bias,
the sampled batch is not representative for the distribution of the unlabeled data [?]. On the counterside, while
diversity-based methods allow for compensation of this problem, they results in an increasing computational
complexity. While [14] and [?] are of the opinion that queries should be based on diversity, it has been shown
that these do not always show better performance [27]. The applicability heavily depends on the selected batch
size for which smaller batch sizes perform more favourable for uncertainy-methods while larger batches are
in favour of diversity. Further, the question of choosing the appropriate query strategy heavily depends its
computational complexity while being considered low in comparison to the labeling budget.

**Human in the loop** as an iterative manner. However, in practice this becomes infeasible. [?] proposed a framework,
humans are able to annotate clusters, reducing the number of interactions required.


# 8 Conclusion
Within the survey, a synopsis of current progresses and open questions within deep active learning for spatio-temporal data was given. While current works on fusing active learning with deep learning has been on a rare sight, most of the studies have centered on uncertainty-based sampling. Further, while diversity-based sampling is an active field of research, we point out that the cost of the respective acquistion function should be taken into regard. 

The main research currently done focuses on optimization of the query strategy. This is largely persuaded that the acquistion function is directly linked to the cost of labeling. For this there is an active discussion of persuing an uncertainty or diversity-based approach. 

Furthermore, hybrid approaches compensates for each of their weaknessess have be gained recent attention. We persue the establishment of uncertainty based methods since the budget compared to labeling is of primer importance. However, diversity methods are worthwhile mentioning, opening new capabilites for realistically capturing the feature distribution. Thus, this will pose novel challenges for the integration of both approaches.