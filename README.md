# Unprobe aka inverse probing

## Probing

Probing refers to decoding some information from the activation patterns of a neural model. For example we can probe a spoken language model (SLM) for speaker identity, for phoneme information, for encoding of words, or for syntactic structure. Probing works well for comparing the ease of decoding the same type of information across layers of a model, or across different models. It is not suitable for comparing across different types of information encoded in the same layer. For example we can probably predict speaker identity from an SLM with close to 100% accuracy, and phoneme information with a much lower accuracy. However we cannot conclude from this that most of the information encoded in the activation patterns in this specific layer is dedicated to encoding speaker identity.

## Unprobing

Unprobing, in contrast, is the reverse: it refers to reconstructing activation patterns from some information, or more typically from a combination of different types of information. In our example of a spoken language we fit a multivariate regression model with the various types of information of interest as predictors: speaker identity, phonemes, words and syntax. The prediction targets are the activation patterns. With this approach we can quantify which predictor is the most informative for reconstructing the activations and compare them to each other. This approach also makes it easier to account for correlations between types of information. For example in a probing approach we may find that we can decode syntax quite well, but this may simply be due to the fact that the activation patterns encode words, and that words are themselves highly predictive of syntax.

In the unprobing setting we would have both words and syntax as predictors, and if syntax doesn't contribute to accuracy over and above words, we would conclude that it's not directly encoded.
