## _ConstrastiveLossFunction.py_
    * A paper based loss function for the training process
## _ds_loader.py_ and _SiameseBundler.py_
    * Load dataset from disk and create bundles/tuples for training. (The Siamese Bundler will be in a never ending loop when less than 4 subjects with images are in the dataset. Does not make a lot of sense to train a network then anyway ;-) )
## _ds_transformations.py_
    * Methods in this script act as collection of strategies that are applied to the images. (Different strategies for training, testing and verification)
    * Normalize and UnNormalize information is coded here as well. Usual values extracted from imageNet trainings are used
## _Main_Training.ipynb_
    * This notebook contains all settings for the training process. Running this will load the dataset and start the training