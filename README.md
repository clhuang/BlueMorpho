# BlueMorpho #

Author: Charlotte Chen, Calvin Huang, Brian Shimanuki, Karthik Narasimhan (karthikn@csail.mit.edu)

### Unsupervised Discovery of Morphological Chains ([TACL 2015](https://tacl2013.cs.columbia.edu/ojs/index.php/tacl/article/view/458)) ###

* A model for unsupervised morphological analysis that integrates orthographic and semantic views of words.
* Exteds Narasimhan's work with additional features, supervised learning, and (not very good) dual-language analyis.
* Based on ([Morphochain](https://github.com/karthikncode/MorphoChain))
* Model consistently outperforms three state-of-the-art baselines on the task of morphological segmentation on Arabic, English and Turkish.

### Download ###
You can clone the repository and use the *master* branch (default) for the latest code.

### Configuration ###
Most parameters in the model can be changed in the file params.properties

### Word Vectors ###
A good tool to produce your own vectors from a raw corpus is [word2vec](https://code.google.com/p/word2vec/). You can also use any pre-existing vectors as long as they satisfy the format as specified in FORMATS.txt.

### Contact ###
Please use the issue tracker or email me if you have any questions/suggestions.
