from torchvision import transforms
from training import MyTransforms


# setting mean and std for normalization
norm_mean = [0.485, 0.456, 0.406] # imageNet mean
# [0.49139968, 0.48215841, 0.44653091] # original DLBIO
norm_std=[0.229, 0.224, 0.225] # imageNet std
# [0.24703223, 0.24348513, 0.26158784] # original DLBIO

normalize = transforms.Normalize( mean=norm_mean, std=norm_std )


def get_transform(dict_key, is_small):

    # dictionary to access different transformation methods
    transform_dict = {
        'train' : transforms_train( get_resize(is_small=is_small) ),
        'valid_and_test' : transforms_valid( get_resize(is_small=is_small) ),
        'size_only' : None
    }
    assert dict_key in transform_dict, "The string "+str(dict_key)+" was not found in dictionary"
    
    return transform_dict[dict_key]

# Returns boolean decision of small or bigger
def get_resize(is_small):
    if is_small: return 150, 100
    else: return 280, 230


# chosen transformations on the training data
def transforms_train(img_shape):
    return transforms.Compose([
        MyTransforms.RandomScaleWithMaxSize(img_shape, 0.8),
        transforms.RandomAffine(degrees=15),
        MyTransforms.MyRandomCrop(crop_ratio=0.1, b_keep_aspect_ratio=True),
        transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.2, hue=0.02),
        #transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.Resize(img_shape),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ])

# chosen transformations on the valid and test data and for later authentication prozess
def transforms_valid( img_shape ):
        return transforms.Compose([
            transforms.Resize(img_shape),
            transforms.ToTensor()
            ])

class UnNormalize(object):
    def __init__(self):
        self.mean = norm_mean
        self.std = norm_std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor