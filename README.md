# Decoding and encoding probes.

## Probes

Probing is term used informally in machine learning to refer to a variety of techniques which find a relation or mapping between model representations (such as layer activation patterns in a neural network) and an external reference representations of the stimuli processed by the model ([Ivanova et al 2021](https://arxiv.org/abs/2104.08197)). The canonical probe is a simple model trained to decode the reference variable from model representations. We can call this type of probe an _decoding _probe_, while a map in the reverse direction would be an encoding probe. It also possible for a probe to be undirected: for example Representational Similarity Analysis (RSA) is a second-order technique which is symmetrical with regards to the two representational spaces.

## Decoding probe
As an example we can use a decoding probe to probe a spoken language model (SLM) for speaker identity, for phoneme information, for encoding of words, or for syntactic structure. Probing works well for comparing the ease of decoding the same type of information across layers of a model, or across different models. It is not suitable for comparing across different types of information encoded in the same layer. For example we can probably predict speaker identity from an SLM with close to 100% accuracy, and phoneme information with a much lower accuracy. However we cannot conclude from this that most of the information encoded in the activation patterns in this specific layer is dedicated to encoding speaker identity.

## Encoding probe 

An encoding probe, in contrast, is the reverse: it refers to reconstructing activation patterns from some information, or more typically from a combination of different types of information. In our example of spoken language we fit a multivariate regression model with the various types of information of interest as predictors: speaker identity, phonemes, words and syntax. The prediction targets are the activation patterns. With this approach we can quantify which predictor is the most informative for reconstructing the activations and compare them to each other. This approach also makes it easier to account for correlations between types of information. For example in a probing approach we may find that we can decode syntax quite well, but this may simply be due to the fact that the activation patterns encode words, and that words are themselves highly predictive of syntax.

In the encoding probe setting we would have both words and syntax as predictors, and if syntax doesn't contribute to accuracy over and above words, we would conclude that it's not directly encoded.
