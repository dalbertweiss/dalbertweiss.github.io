---
layout: post
comments: true
title: "Capabilities of deep learning models to generalize."
date: 2020-01-27 23:45:13 -0400
background: '/img/posts/02.jpg'
---


While the of implementation and training deep learning models becomes more easy to accomplish the generalization of deep learning still remains an active research field. This specificially concerns the use of stochastic gradient descent (SGD) which by Schmidthuber has often been assosciated with a flat minima during training [1]. However, this has largely been dismissed by <a href="https://arxiv.org/pdf/2103.06219.pdf/">[2]</a>.

During this article I would like to discuss some approaches that has arised for shedding light behind understanding the generalization capabilities of deep learning models. This article is organized the following way:


# General Understanding
Deep learning models are known to have tens of thousands of parameters leading to a larger search space during model optimization. 
The question aring here is how meaniful the label is in allowing to generealize well. The paper of <a href="https://arxiv.org/pdf/1611.03530.pdf">[3]</a> tried to equip the input images with completely random labels, showing that the neural network can easily fit to the training data. They show that with the effective capacity of the model being large enough, the neural network is able to
adapt to random labels.

While this experiment indicates that generalization of complex models is not as straighforward since these are vacuous and non-predictive the need for need for novel fundamentals arises <a href="https://ai.googleblog.com/2021/03/a-new-lens-on-understanding.html">[4]</a>.


To encounter this Prof. Naftali Tishby, hold an interesting <a href="https://www.youtube.com/watch?v=XL07WEc2TRI">talk</a> where relates the capabilities to generalize well with informationtheoretic theory.
https://adityashrm21.github.io/Information-Theory-In-Deep-Learning/


# Classical approaches

<!---Occam’s Razor
Minimum Description Length principle
Kolmogorov Complexity
Solomonoff’s Inference Theory---->
**test**

<div class="tip" markdown="1">Have **fun!** test *test*</div>

<h2 class="section-heading">Classical approaches</h2>
<h3 class="subsection-heading">Flat Minima</h3>
In 1997 Sepp Hoch__ and Schmidthuber studied the local, geometrical properties 
during optimization of deep learning with the assumption that reason for generalization
well is correlated with converging to a flat minima. This hypothesis is supposed to come
from the fact that the flat minima tends to be more stable.




<blockquote class="blockquote">The dreams of yesterday are the hopes of today and the reality of tomorrow. Science has not yet mastered prophecy. We predict too much for the next year and yet far too little for the next ten.</blockquote>

<p>Spaceflights cannot be stopped. This is not the work of any one man or even a group of men. It is a historical process which mankind is carrying out in accordance with the natural laws of human development.</p>

<h2 class="section-heading">Reaching for the Stars</h2>

<p>As we got further and further away, it [the Earth] diminished in size. Finally it shrank to the size of a marble, the most beautiful you can imagine. That beautiful, warm, living object looked so fragile, so delicate, that if you touched it with a finger it would crumble and fall apart. Seeing this has to change a man.</p>

<img class="img-fluid" src="https://source.unsplash.com/Mn9Fa_wQH-M/800x450" alt="Demo Image">
<span class="caption text-muted">To go places and do things that have never been done before – that’s what living is all about.</span>

<p>Space, the final frontier. These are the voyages of the Starship Enterprise. Its five-year mission: to explore strange new worlds, to seek out new life and new civilizations, to boldly go where no man has gone before.</p>

<p>As I stand out here in the wonders of the unknown at Hadley, I sort of realize there’s a fundamental truth to our nature, Man must explore, and this is exploration at its greatest.</p>

<p>Placeholder text by <a href="http://spaceipsum.com/">Space Ipsum</a>. Photographs by <a href="https://unsplash.com/">Unsplash</a>.</p>
