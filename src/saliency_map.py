import saliency.core as saliency
import PIL.Image
from matplotlib import pylab as P
import numpy as np
import torch
from torchvision import transforms
from model_local import local_model

model = local_model("/scratch/scratch6/akansh12/DeepEXrays/local_label/0.891597_.pth")

def ShowImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im)
    P.title(title)

def ShowGrayscaleImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
    P.title(title)

def ShowHeatMap(im, title, ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im, cmap='inferno')
    P.title(title)

def LoadImage(file_path):
    im = PIL.Image.open(file_path)
    im = im.resize((254, 254))
    im = np.asarray(im)
    return im

def PreprocessImages(images):
    # assumes input is 4-D, with range [0,255]
    #
    # torchvision have color channel as first dimension
    # with normalization relative to mean/std of ImageNet:
    #    https://pytorch.org/vision/stable/models.html
    images = np.array(images)
    images = images/255
    images = np.transpose(images, (0,3,1,2))
    images = torch.tensor(images, dtype=torch.float32)
    images = transformer.forward(images)
    return images.requires_grad_(True)

transformer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
class_idx_str = 'class_idx_str'
def call_model_function(images, call_model_args=None, expected_keys=None):
    images = PreprocessImages(images)
    target_class_idx =  call_model_args[class_idx_str]
    output = model(images)
    m = torch.nn.Softmax(dim=1)
    output = m(output)
    if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
        outputs = output[:,target_class_idx]
        grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))
        grads = torch.movedim(grads[0], 1, 3)
        gradients = grads.detach().numpy()
        return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
    else:
        one_hot = torch.zeros_like(output)
        one_hot[:,target_class_idx] = 1
        model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)
        return conv_layer_outputs


def xrai_masks(path2img, model):
    im_orig = LoadImage(path2img)
    im_tensor = PreprocessImages([im_orig])
    # Show the image
    # ShowImage(im_orig)

    predictions = model(im_tensor)
    predictions = predictions.detach().numpy()
    prediction_class = (predictions[0]>0.5).astype('int')
    call_model_args = {class_idx_str: prediction_class}

    im = im_orig.astype(np.float32)



    xrai_object = saliency.XRAI()

    xrai_attributions = xrai_object.GetMask(im, call_model_function, call_model_args, batch_size=20)

    return xrai_attributions
#     ShowHeatMap(xrai_attributions, title='XRAI Heatmap', ax=P.subplot(ROWS, COLS, 2))




















