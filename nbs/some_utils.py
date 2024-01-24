# Import necessary libraries
from fastai.imports import *
from fastai.torch_core import *
from fastai.learner import *
from fastai.vision.all import *
from palettable.colorbrewer.diverging import RdYlBu_10_r
from typing import List, Tuple, Callable, Union, Optional, Any, Dict
from torchvision.utils import make_grid
from fastai_amalgam.interpret.all import *

##########################################################
## Function to plot metrics after learning. 
## Based on https://forums.fast.ai/t/plotting-metrics-after-learning/69937/3
##########################################################

@patch
@delegates(subplots)
def plot_metrics(self: Recorder, nrows=None, ncols=None, figsize=None, endnames=-1, **kwargs):
    # Stack metrics values
    metrics = np.stack(self.values)
    # Get metric names
    names = self.metric_names[1:endnames]
    n = len(names) - 1
    # Calculate number of rows and columns for subplots
    nrows, ncols = calculate_subplot_dimensions(n, nrows, ncols)
    # Set default figure size if not provided
    figsize = figsize or (ncols * 6, nrows * 4)
    # Create subplots
    fig, axs = create_subplots(nrows, ncols, figsize, metrics, names, **kwargs)
    plt.show()

# Function to calculate subplot dimensions
def calculate_subplot_dimensions(n, nrows, ncols):
    if nrows is None and ncols is None:
        nrows = int(math.sqrt(n))
        ncols = int(np.ceil(n / nrows))
    elif nrows is None: nrows = int(np.ceil(n / ncols))
    elif ncols is None: ncols = int(np.ceil(n / nrows))
    return nrows, ncols

# Function to create subplots
def create_subplots(nrows, ncols, figsize, metrics, names, **kwargs):
    fig, axs = subplots(nrows, ncols, figsize=figsize, **kwargs)
    axs = [ax if i < len(names) - 1 else ax.set_axis_off() for i, ax in enumerate(axs.flatten())][:len(names) - 1]
    for i, (name, ax) in enumerate(zip(names, [axs[0]] + axs)):
        ax.plot(metrics[:, i], color='#1f77b4' if i == 0 else '#ff7f0e', label='valid' if i > 0 else 'train')
        ax.set_title(name if i > 1 else 'losses')
        ax.legend(loc='best')
    return fig, axs



##########################################################
## CAM and grad-CAM for multiclass, based on fastai_amalgam
## (https://github.com/Synopsis/amalgam)
##########################################################
        
# Convert image to array-like
def convert_image_to_arraylike(img:Image.Image, as_array=False, as_tensor=True):
    img = np.array(img).transpose(2,0,1)
    if as_array: return img
    if as_tensor: return torch.from_numpy(img)

# Create a grid of images
def create_image_grid(images : Union[List[str], List[PILImage], List[Image.Image]],
                      img_size : Union[tuple, float, None] = (480,270),
                      ncol : int=8
                      ) -> Image.Image:
    if not isinstance(images, (list,L)): raise TypeError(f"Expected a list of (paths,PILImage,PIL.Image.Image) objects, got {type(images)} instead")
    img_list = prepare_image_list(images, img_size)
    grid_array = make_grid(img_list, nrow=ncol)
    grid_array = grid_array.numpy().transpose((1,2,0))
    return Image.fromarray(grid_array)

# Prepare a list of images
def prepare_image_list(images, img_size):
    if isinstance(images[0], (str,Path)):
        img_list = [open_image(f, as_tensor=True, size=img_size) for f in images]
    elif isinstance(images[0], (PILImage, Image.Image)):
        img_list = images
        if img_size is not None:
            img_list = [img.resize((img_size)) for img in img_list]
        img_list = [convert_image_to_arraylike(img) for img in img_list]
    return img_list


# Function to plot decoded image
def plt_decoded(learn, x, ctx, cmap=None):
    'Processed tensor --> plottable image, return `extent`'
    x_decoded = TensorImage(learn.dls.train.decode((x,))[0][0])
    extent = (0, x_decoded.shape[1], x_decoded.shape[2], 0)
    x_decoded.show(ctx=ctx, cmap=cmap)
    return extent

# Function to plot Grad-CAM map over image
def plot_gcam(learn, img:PILImage, x:tensor, gcam_map:tensor,
              full_size=True, alpha=0.6, dpi=100,
              interpolation='bilinear', cmap=None, gcam_cmap='magma', **kwargs):
    'Plot the `gcam_map` over `img`'
    fig,ax = plt.subplots(dpi=dpi, **kwargs)
    if full_size:
        extent = (0, img.width,img.height, 0)
        show_image(img, ctx=ax, cmap=cmap)
    else:
        extent = plt_decoded(learn, x, ax, cmap=cmap)

    show_image(gcam_map.detach().cpu(), ctx=ax,
               alpha=alpha, extent=extent,
               interpolation=interpolation, cmap=gcam_cmap)

    return plt2pil(fig)


def gradcam(self: Learner,
            item: Union[PILImage, os.PathLike],
            target_layer: Union[nn.Module, Callable, None] = None,
            labels: Union[str,List[str], int,List[int], None] = None,
            full_size=False, show_original=False, img_size=None, alpha=0.5,
            cmap = None,
            gcam_cmap = RdYlBu_10_r.mpl_colormap,
            font_path=None, font_size=None, grid_ncol=4,
            **kwargs
           ):
    """Plot Grad-CAMs of all specified `labels` with respect to `target_layer`
    Key Args:
    * `item`: a `PILImage` or path to a file. Use like you would `Learner.predict`
    * `target_layer`: The target layer w.r.t which the Grad-CAM is produced
                      Can be a function that returns a specific layer of the model
      or also a direct reference such as `learn.model[0][2]`. If `None`,
      defaults to `learn.model[0]`
    * `labels`: A string, int index, or list of the same w.r.t which the Grad-CAM
                must be plotted. If `None`, the top-prediction is plotted if the
      model uses a Softmax activation, else it must be specified.
    * `show_original`: Show the original image without the heatmap overlay
    * `font_path`: (Optional, recommended) Path to a `.ttf` font to render the text
    * `font_size`: Size of the font rendered on the image
    * `grid_ncol`: No. of columns to be shown. By default, all maps are shown in one row
    """
    dl = self.dls.test_dl([item])
    x = detuplify(first(dl))


    if not isinstance(labels, list): labels=[labels]
    if isinstance(item, PILImage): img = item
    else: img = PILImage.create(item)
    if img_size is not None: img=img.resize(img_size)
    if grid_ncol is None: grid_ncol = 1+len(labels) if show_original else len(labels)

    gcams = defaultdict()

    results = []
    for label in labels:
        grads, acts, preds, _label = compute_gcam_items(self, x, label, target_layer)
        gcams[label] = compute_gcam_map(grads, acts)
        preds_dict = {l:pred for pred,l in zip(preds, self.dls.vocab)}
        pred_img = plot_gcam(self, img, x, gcams[label], full_size=full_size, alpha=alpha, cmap=cmap, gcam_cmap=gcam_cmap)
        pred_img.draw_labels(f"{_label}: {preds_dict[_label]* 100:.02f}%",
                             font_path=font_path, font_size=font_size, location="top")
        results.append(pred_img)
    if show_original:
        img = img.resize(results[0].size)
        results.insert(0, img.draw_labels("Original", font_path=font_path, font_size=font_size, location="top"))
    return create_image_grid(results, img_size=None, ncol=grid_ncol)



