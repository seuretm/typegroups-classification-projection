# typegroups-classification-projection

## About data

To run this code, you will need to have document images in the following
structure:

```
a base directory
  Containing at least one sub-directory
    Containing at least one image
```

The sub-directories names correspond to the class you attributed to the
document images they contain.


## Programs descriptions

### feature_extraction
Extracts values produces by the penultimate layer of the neural network
from a given amount of patches from each page, and stores them into a
file.

> python3 feature_extraction.py

### project_features
Computes a t-SNE from previously computed features, plots it, and stores
it as an image.

### patchwise_classification
Produces classification scores for individual patches can be obtained
with patchwise_classification, and stores the whole as an html page.
Note that a folder called "img" will be needed, otherwise the execution
will fail.
