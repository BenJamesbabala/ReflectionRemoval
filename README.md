# ReflectionRemoval

The repo was developed and tested on a Nvidia RTX 2070 on Windows 10. I also tested test.py on Mac. See the conda environment files in env folder.
To run plot-tsne.py, make sure you installed scikit-learn.

Download VGG-19 model [__here__](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat) and put it to the root folder.

## Training

```
python train.py
```
See more parameters in train.py.

## Testing

```
python test.py
```
See more parameters in test.py.

## Reference

Part of the code is based on https://github.com/ceciliavision/perceptual-reflection-removal
