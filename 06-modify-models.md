---
title: "Modify standard DL models for geospatial data and spatial context"
teaching: 10 # teaching time in minutes
exercises: 2 # exercise time in minutes
---


:::::::::::::::::::::::::::::::::::::: questions

- How may geospatial and Earth Observation data differ from natural images?
- How do these changes potentially affect adoption of core Deep Learning techniques?
- How to embed spatial and temporal context in deep learning models?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Explain why geospatial and Earth Observation data when treated as simple images may not be able to achieve the best performance out of Deep Learning models due to differences in data characteristics.
- Describe at least three strategies of injecting geospatial information into Deep Learning model performance.
- Build a data loading pipeline that handles multi-band, multi-resolution EO rasters.
- Compare trade-offs of each strategy in terms of performance vs data efficiency, compute cost, implementation complexity.
- Modify standard convolution or transformer-based Deep Learning architectures to accept multispectral or auxiliary-variable inputs.
- Judge for a given problem which strategy is the most appropriate.
::::::::::::::::::::::::::::::::::::::::::::::::


::::::::::::::::::::::::::::::::::::: callout

### Note
It is highly recommended that you pre-view episodes in light-mode.
::::::::::::::::::::::::::::::::::::::::::::::::

## Introduction

In this lesson, you will learn how the unique characteristics of geospatial, specifically Earth Observation (EO), data affect the adoption of core Deep Learning (DL) techniques.

We begin by introducing what makes geospatial/EO data different and where those differences limit the direct applicability of standard DL approaches. From there, we focus on the practical realities of working with EO datasets for a given DL task. We then turn to concrete DL method adaptations for handling geospatial data: including pretrained models, extended input channels, and lightweight spatial attention.

We close by discussing the trade-offs between these approaches and how to judge which is the right fit for a given problem.

## Deep Learning with Geospatial Data


Deep Learning (DL) is a powerful tool to process and extract meaningful information from large raw data at scale. It is widely used across domains to extract features, capture patterns, segment regions, detect objects, and comprehend sequences. When applied to geospatial data, specifically Earth Observation (EO) data, DL presents significant opportunities to automate the analysis and modeling of satellite and aerial imagery at scale.

Common DL applications for geospatial data include:

- Land use and land cover classification and change detection.
- Deforestation, wild-fire detection and mapping.
- Building footprint extraction and urban-sprawl monitoring.
- Crop yield prediction and agricultural monitoring.
- Disaster impact assessment.
- Maritime surveillance (ship detection, oil spill tracking).
- Many more.

## Unique Characteristics of Geospatial Data in Deep Learning Domain

However, most standard DL architectures are designed for common data modalities, such as natural photographs or language. Applying these out-of-the-box models to EO data can yield suboptimal results because EO data possesses unique characteristics that demand it be treated as a distinct modality [(Rolf et al., 2024)](https://doi.org/10.48550/arXiv.2402.01444).

These unique EO characteristics include:

- **Varying Spatial Scales:** Target sizes span a logarithmic scale, from sub-meter (tree) to over a kilometer (field).
- **Overhead Perspective:** Targets are captured from a top-down view, and lack a “natural” orientation, unlike natural images.
- **Temporal Dynamics:** Temporal patterns also span a logarithmic scale, tracking changes from hours to decades.
- **Spectral Depth:** Data is often multi- or hyperspectral, capturing wavelengths far beyond standard RGB.
- **Radiometric Resolution:** Data is measured at a higher precision, frequently exceeding standard 8-bit depth.
- **Non-Optical Modalities:** EO encompasses more than visual imagery: it integrates structural and active sensor data, such as Digital Elevation Models (DEM), LiDAR, and Synthetic Aperture Radar (SAR).
- **Global Coverage:** Datasets inherently span the entire globe, introducing immense geographic and environmental diversity.

While general DL techniques can be adopted directly for baseline tasks, unlocking their full performative capacity requires neural architectures tailored to the specific complexity of geospatial information. Applying DL to EO data is not just a data processing task; it is a unique integration challenge with no equivalents in conventional computer vision. To be truly effective, models must be designed to handle the heterogeneity, high dimensionality, and spatial autocorrelation inherent in remote sensing, GIS, and spatial analytics.


::::::::::::::::::::::::::::::::::::: challenge

## Challenge: Name the key differences between natural and EO image data?


![Images of ten different types of cats from ImageNet dataset. [Source](https://doi.org/10.3390/app11156963?urlappend=%3Futm_source%3Dresearchgate.net%26utm_medium%3Darticle)](https://www.mdpi.com/applsci/applsci-11-06963/article_deploy/html/images/applsci-11-06963-g001.png){alt='Collage of pictures of cats and (potentially) racoons.'}

![Satellite images of the same location [Source: Rolf et al., 2024](https://doi.org/10.48550/arXiv.2402.01444)](https://github.com/GeospatialML/geo-ml-modelling-lesson/blob/main/episodes/images/eo_modalities_Rolf.png?raw=true){alt='Collage of satellite images and products, depicting varied spatial resolutions, temporal dimension, information content.'}

:::::::::::::::::::::::: solution

## Answer

Natural images are object centric, with virtually identical semantics. While satellite images of the same location can vary widely and capture different semantic meanings. They depend on factors like spatial resolution and cropping extent, temporal
dimension, and satellite mission or instrument.

:::::::::::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::

## Integration of Geospatial Data for DL Pipelines

If you have tried to experiment with EO images in a standard computer vision pipeline, you have probably encountered several hurdles: the images had large file sizes and were not in an easy-to-open format (e.g. with PIL's `Image`), images themselves had unconventional dimensions: large width and height, too few channels (e.g. single-band Digital Elevation Model rasters) or too many channels relative to native RGB images,while the pixel values were not stored as standard 8-bit unsigned integers. Integrating multiple modalities could introduce an additional hiccup, since inputs may differ in spatial resolution, image size, and coordinate reference system. Further, at inference, you need to obtain predictions for entire or even multiple scenes producing one consolidated map. This requires obtaining predictions from overlapping tiles and singing them into georeferenced maps.

As highlighted in the previous section, these mismatches and implementation challenges stem from EO data having quite different characteristics from natural images, and thus require converting raw sensor measurements into DL analysis-ready data formats.

This includes choices around projection and sampling strategy, scaling, choice of spectral bands and input modalities, and the (optional) alignment of the various data modalities. These choices should support workflows that stay robust to differing coordinate systems, resolutions, and sensor characteristics.

::::::::::::::::::::::::::::::::::::: callout

### Note
All the choices leading up to a DL-ready dataset introduce effects and uncertainties that propagate through the entire modeling pipeline. These choices *can* be reversed, but at a cost. Therefore, these choices should be treated as part of the modeling pipeline and they should be well documented.

::::::::::::::::::::::::::::::::::::::::::::::::

In theory, each EO data point is a multi-dimensional image patch with $W \times H \times T \times C$ dimensions, where $W$ is the width, $H$ is the height, $T$ is the timestep, and $C$ is the number of spectral channels. We will walk through each dimension in turn.

### 1. Spatial dimension
**Working with patches**. Models are rarely trained on full scenes directly, as these scene images are too large to be passed through most neural networks. Therefore, scenes are cropped into fixed-size patches for training. Patch size is itself a modeling choice: it trades off spatial context (larger patches capture more surrounding structure) against memory/compute cost and the number of training samples you can extract per scene.

**Inference over full scenes** at inference time: the same patch size is typically re-applied via a sliding window over the full scene, with overlapping tiles to avoid edge artifacts; predictions are then merged (e.g., averaged in overlap regions) and written back out as a single georeferenced raster.
TODO: Further patching in ViT

### 2. Temporal dimension

There are several options on how to deal with the temporal dimension. Sialelli et al. (2026) summarized them as the following options:

- **Time-series / time-window**: treating EO data either as per-pixel time series, or as video-like sequences of images over a time window.
- **Composites**: collapsing a time window into a single representative image, e.g. taking the per-pixel median to suppress clouds and transient noise.
- **Single time-step**: treating the data as an instantaneous snapshot, ignoring temporal context entirely.

### 3. Spectral dimension

**Radiometric resolution**

Radiometric pre-processing of raw data matters and should not be treated as an afterthought. Raw sensor digital numbers are typically scaled to a physical measurement (e.g. reflectance). Note the distinction between Top-of-Atmosphere and Bottom-of-Atmosphere reflectance products, as they are not interchangeable. For Sentinel-2, this typically means dividing digital numbers by 10,000 to recover reflectance values in [0, 1].

DL pipelines sometimes substitute alternative divisors: normalizing to the training data's own distribution, or applying band-specific scaling factors. Alternatively, if you are building on top of pre-trained models, you may need to match the pretrained model's expected input distribution. 

All this pre-processing also includes the dtype conversion. When reusing a pretrained RGB model, data sometimes needs to be converted to 8-bit unsigned integers (`uint8`) to match what that model was trained on.

When in doubt on how to pre-process, scale or normalize pixel values: read the sensor/product documentation rather than assuming a standard scaling convention.

::::::::::::::::::::::::::::::::::::: callout

### Standardisation and normalisation
Scaling inputs to roughly [-1, 1] generally aids training stability: large input values can push activations of functions such as sigmoid and tanh into their saturation regime, producing diminished gradients and slower convergence (Ioffe and Szegedy, 2015).

::::::::::::::::::::::::::::::::::::::::::::::::

**Spectral coverage**

*Channel selection*: when using a pretrained RGB model, you may choose to keep only the 3 bands closest to that model's native RGB input, discarding or separately handling the rest.

However, as discussed previously, using additional EO data can guide the model toward more accurate predictions. Rather than restricting yourself to 3 bands, you may opt to keep the full multispectral stack, or add channels beyond the sensor's native bands entirely.

*Other modalities*: incorporating derived indices (e.g. NDVI) or entirely different modalities (DEM, SAR) alongside optical bands. When training patches combine data from multiple sensors, the inputs will generally differ in native projection, pixel spacing, and temporal sampling. The most common solution is to reproject and resample all modalities onto a shared grid and resolution before patch extraction, so each training sample is spatially aligned.

An alternative is to preserve each modality at its native resolution and delegate alignment to the model architecture itself — avoiding information loss at the cost of greater architectural complexity. Typical designs use modality-specific encoder branches, each operating at its source's native resolution, with representations fused at a shared spatial scale via learned upsampling, cross-attention, or feature concatenation after a spatial-alignment layer.

## TorchGeo: Custom and Curated Datasets

TorchGeo is a modular, scalable Python framework for integrating geospatial data into DL workflows. It is built on top of PyTorch, extending it to handle spatio-temporal geospatial data. TorchGeo addresses the full pipeline and can be split into a data component and a modeling component. In this section, we focus on TorchGeo's dataset classes. The next section covers the modeling side.

TorchGeo's dataset classes provide a solid starting point, giving access to already curated, established DL-ready datasets from the DL-for-EO community. At the same time, TorchGeo's dataset structure can serve as a template for packaging your own dataset for reuse and collaboration.

TorchGeo datasets bake in the sample pre-processing that implements the techniques discussed above, so the data arrives ready for DL training or evaluation pipelines.

::::::::::::::::::::::::::::::::::::: callout

### Curated datasets in TorchGeo
The available curated datasets are listed [here](https://docs.torchgeo.org/en/stable/api/datasets.html).

- These datasets are the product of individual or community efforts, and many serve as established benchmarks/baselines in the field.
- They lower the barrier to entry for DL practitioners, who would otherwise face the non-trivial task of assembling training-ready inputs from raw or derived EO products themselves.
- The trade-off is a loss of control: adopting an existing curated dataset means inheriting its creators' design choices, which may not match the requirements of your downstream task.
::::::::::::::::::::::::::::::::::::::::::::::::

In this episode, we will work with EuroSAT100, a subset of the EuroSAT dataset, which is an established image classification dataset. It is composed of multispectral images from the Sentinel-2 satellites and has 10 class labels (see figure bellow).

![Example images for the 10 classes in the EuroSAT dataset from Helber et al. (2019)](https://github.com/GeospatialML/geo-ml-modelling-lesson/blob/main/episodes/images/eurosat_Helber.png?raw=true){alt='Collage of satellite images subdivided based on classes: Annual Crop, Forest, Herbaceous Vegetation, Highway, Industrial Buildings, Pasture, Permanent Crop, Residential Buildings, River, Sea & Lake.'}

This dataset can be loaded from TorchGeo datasets:

```python
from torchgeo.datasets import EuroSAT100, EuroSAT

train_dataset = EuroSAT100(
    root='data/',
    split='train',
    download=True
)
# To use the full dataset replace EuroSAT100 with EuroSAT

```

::::::::::::::::::::::::::::::::::::: challenge

## Challenge: Create validation and test dataset

Following the example above create `val_dataset` and `test_dataset` dataset objects.

:::::::::::::::::::::::: solution

## Answer

```python

val_dataset = EuroSAT100(
    root='data/',
    split='val',
    download=True
)

test_dataset = EuroSAT100(
    root='data/',
    split='test',
    download=True
)
```
:::::::::::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::

You can also sub-select only the RGB bands:

```python
train_dataset_rgb = EuroSAT100(
    root='data/', 
    split='train', 
    download=True, 
    bands=('B04', 'B03', 'B02')
)
```

The images in the dataset are in the original Sentinel-2 scaling. Therefore, we need to rescale the dataset back to reflectance range of $[0, 1]$ by dividing by $10^4$.

```python
from torchvision.transforms import v2

preprocess = v2.Normalize(mean=[0.0], std=[10000.0])
```

::::::::::::::::::::::::::::::::::::: challenge

## Challenge: Inspect pixel value ranges

Knowing that `dataset[0]` returns the first entry in the dataset, inspect minimum and maximum values before and after applying `preprocess(dataset[0]['image'])`

:::::::::::::::::::::::: solution

## Answer

```python
sample = dataset[0]['image']
print(f"Before preprocessing: min={sample.min()}, max={sample.max()}")
sample = preprocess(sample)
print(f"After preprocessing: min={sample.min()}, max={sample.max()}")
```
:::::::::::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: callout

### Augmentations
Augmentations let us virtually increase dataset size and help train more generalizable models. However this is only true when augmentations introduce variation that occurs naturally in the data.

Hopkins et al. (2025) found that "augmentation techniques designed for natural images should not be applied to satellite imagery without careful consideration". Their results suggest that standard natural-image techniques, particularly photometric augmentations, do not translate well to the satellite domain, while geometric operations remain broadly beneficial.

- Geometric transformations such as flipping and rotation tend to be safe (with exceptions), since overhead imagery rarely has an inherent "up" direction — unlike, say, a natural photo of a tree, which reads as wrong upside-down. Random crop or zoom need more care: because satellite imagery has a fixed, known ground sampling distance, arbitrary rescaling can distort the physical meaning of the pixels. Whether a specific geometric transformation is acceptable depends on the end task.
- From photometric transformations, adding noise is commonly used and can be beneficial.
- For temporal augmentations: dropping time-steps to mimic cloud occlusion, or masking out missing data modalities.

You can implement augmentations as follows:

```python

from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode

transforms = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation([0, 360], interpolation=InterpolationMode.BILINEAR),
    v2.GaussianNoise(mean = 0.0, sigma = 0.1)
    
])
```
::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: callout

### Adding Indices in TorchGeo

You can append indices to any dataset as follows:

```python
from torchgeo.transforms import AppendNDVI

add_ndvi =  AppendNDVI(index_nir=7, index_red=3) # it can be stacked with other transformations/augmentations

```

You can find more indices [here](https://docs.torchgeo.org/en/v0.2.0/api/transforms.html)

::::::::::::::::::::::::::::::::::::::::::::::::

Now let's create dataloaders for model training and evaluation. Note that validation and test datasets should not be shuffled, contain data augmentations (except for data preprocessing steps).
```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, transforms=v2.Compose([transforms, preprocess]))
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, transforms=preprocess)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, transforms=preprocess)

```

# 



::::::::::::::::::::::::::::::::::::: challenge

## Challenge 

:::::::::::::::::::::::: solution

## Answer


:::::::::::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::




## References 

There is much more about [torchgeo](https://pytorch.org/blog/geospatial-deep-learning-with-torchgeo/) for working with geo-spatial data.

- Robinson & Corley (2026) https://geospatialml.com/posts/torchgeo-iclr-tutorial/
- Sialelli et al. (2026) https://ghjuliasialelli.github.io/ML-EO-Maps/data_selection.html
- GeoWGS84AI  https://www.geowgs84.ai/post/deep-learning-for-geospatial-analysis-best-practices-code-samples and https://www.geowgs84.ai/post/torchgeo-for-beginners-unlocking-ai-in-geospatial-applications
- Hopkins et al. (2025) https://ojs.aaai.org/index.php/AAAI/article/view/35028
- Ioffe and Szegedy (2015) https://arxiv.org/abs/1502.03167
- Rolf et al. (2024) https://doi.org/10.48550/arXiv.2402.01444

::::::::::::::::::::::::::::::::::::: keypoints

- tba
- tba

::::::::::::::::::::::::::::::::::::::::::::::::


:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: instructor

Notes for instructors

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::