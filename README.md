# Mixture of Softmaxes

Keras implementation of Mixture of Softmaxes. This layer is a type of ensenmble method described in
[Breaking the Softmax Bottleneck: A High-Rank RNN Language Model](https://arxiv.org/abs/1711.03953).

I have linked below a few blogs that can do this layer more justice than I can.

## Experiments

I'm planning on testing this layer with a few different architectures and datasets.

For MNIST I have compared the mixture of softmaxes -- where we combine 3 softmaxes -- and just plain softmax and the improvement is around %1 in accuracy. See *MNIST* notebook.

The plan is to play around with *CIFAR-10* and *CIFAR-100* next. I will then move to some actual language models.

## Resources

Some useful references:

- The official implementation can be found [here](https://github.com/zihangdai/mos).
- There two interesting blog posts that are quite enlightening. Found [here](http://smerity.com/articles/2017/mixture_of_softmaxes.html) and [here](https://severelytheoretical.wordpress.com/2018/06/08/the-softmax-bottleneck-is-a-special-case-of-a-more-general-phenomenon/).
- Mixture of Experts layer for Keras is [here](https://github.com/eminorhan/mixture-of-experts).
