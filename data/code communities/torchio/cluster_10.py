# Cluster 10

def _create_categorical_colormap(data: torch.Tensor, cmap_name: str='glasbey_category10') -> tuple[ListedColormap, BoundaryNorm]:
    num_classes = int(data.max())
    mpl, _ = import_mpl_plt()
    colors = [(0, 0, 0), (1, 1, 1)]
    if num_classes > 1:
        from .external.imports import get_colorcet
        colorcet = get_colorcet()
        cmap = getattr(colorcet.cm, cmap_name)
        color_cycle = cycle(cmap.colors)
        distinct_colors = [next(color_cycle) for _ in range(num_classes - 1)]
        colors.extend(distinct_colors)
    boundaries = np.arange(-0.5, num_classes + 1.5, 1)
    colormap = mpl.colors.ListedColormap(colors)
    boundary_norm = mpl.colors.BoundaryNorm(boundaries, ncolors=colormap.N)
    return (colormap, boundary_norm)

def import_mpl_plt():
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError('Install matplotlib for plotting support') from e
    return (mpl, plt)

def plot_volume(image: Image, radiological=True, channel=None, axes=None, cmap=None, output_path=None, show=True, xlabels=True, percentiles: tuple[float, float]=(0, 100), figsize=None, title=None, reorient=True, indices=None, rgb=True, savefig_kwargs: dict[str, Any] | None=None, **imshow_kwargs) -> Figure | None:
    _, plt = import_mpl_plt()
    fig: Figure | None = None
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    if reorient:
        image = ToCanonical()(image)
    is_label = isinstance(image, LabelMap)
    if is_label:
        data = image.data[np.newaxis, -1]
    elif rgb and image.num_channels == 3:
        data = image.data
    elif channel is None:
        data = image.data[0:1]
    else:
        data = image.data[np.newaxis, channel]
    data = rearrange(data, 'c x y z -> x y z c')
    data_numpy: np.ndarray = data.cpu().numpy()
    if indices is None:
        indices = np.array(data_numpy.shape[:3]) // 2
    i, j, k = indices
    slice_x = rotate(data_numpy[i, :, :], radiological=radiological)
    slice_y = rotate(data_numpy[:, j, :], radiological=radiological)
    slice_z = rotate(data_numpy[:, :, k], radiological=radiological)
    if isinstance(cmap, dict):
        slices = (slice_x, slice_y, slice_z)
        slice_x, slice_y, slice_z = color_labels(slices, cmap)
    else:
        boundary_norm = None
        if cmap is None:
            if is_label:
                cmap, boundary_norm = _create_categorical_colormap(data)
            else:
                cmap = 'gray'
        imshow_kwargs['cmap'] = cmap
        imshow_kwargs['norm'] = boundary_norm
    if is_label:
        imshow_kwargs['interpolation'] = 'none'
    elif 'interpolation' not in imshow_kwargs:
        imshow_kwargs['interpolation'] = 'bicubic'
    imshow_kwargs['origin'] = 'lower'
    if not is_label:
        displayed_data = np.concatenate([slice_x.flatten(), slice_y.flatten(), slice_z.flatten()])
        p1, p2 = np.percentile(displayed_data, percentiles)
        if 'vmin' not in imshow_kwargs:
            imshow_kwargs['vmin'] = p1
        if 'vmax' not in imshow_kwargs:
            imshow_kwargs['vmax'] = p2
    spacing_r, spacing_a, spacing_s = image.spacing
    sag_axis, cor_axis, axi_axis = axes
    slices_dict = {'Sagittal': {'aspect': spacing_s / spacing_a, 'slice': slice_x, 'xlabel': 'A', 'ylabel': 'S', 'axis': sag_axis}, 'Coronal': {'aspect': spacing_s / spacing_r, 'slice': slice_y, 'xlabel': 'R', 'ylabel': 'S', 'axis': cor_axis}, 'Axial': {'aspect': spacing_a / spacing_r, 'slice': slice_z, 'xlabel': 'R', 'ylabel': 'A', 'axis': axi_axis}}
    for axis_title, info in slices_dict.items():
        axis = info['axis']
        axis.imshow(info['slice'], aspect=info['aspect'], **imshow_kwargs)
        if xlabels:
            axis.set_xlabel(info['xlabel'])
        axis.set_ylabel(info['ylabel'])
        axis.invert_xaxis()
        axis.set_title(axis_title)
    plt.tight_layout()
    if title is not None:
        plt.suptitle(title)
    if output_path is not None and fig is not None:
        if savefig_kwargs is None:
            savefig_kwargs = {}
        fig.savefig(output_path, **savefig_kwargs)
    if show:
        plt.show()
    return fig

def rotate(image: np.ndarray, *, radiological: bool=True, n: int=-1) -> np.ndarray:
    image = np.rot90(image, n, axes=(0, 1))
    if radiological:
        image = np.fliplr(image)
    return image

def plot_subject(subject: Subject, cmap_dict=None, show=True, output_path=None, figsize=None, clear_axes=True, **plot_volume_kwargs):
    _, plt = import_mpl_plt()
    num_images = len(subject)
    many_images = num_images > 2
    subplots_kwargs = {'figsize': figsize}
    try:
        if clear_axes:
            subject.check_consistent_spatial_shape()
            subplots_kwargs['sharex'] = 'row' if many_images else 'col'
            subplots_kwargs['sharey'] = 'row' if many_images else 'col'
    except RuntimeError:
        pass
    args = (3, num_images) if many_images else (num_images, 3)
    fig, axes = plt.subplots(*args, **subplots_kwargs)
    axes = axes.T if many_images else axes.reshape(-1, 3)
    iterable = enumerate(subject.get_images_dict(intensity_only=False).items())
    axes_names = ('sagittal', 'coronal', 'axial')
    for image_index, (name, image) in iterable:
        image_axes = axes[image_index]
        cmap = None
        if cmap_dict is not None and name in cmap_dict:
            cmap = cmap_dict[name]
        last_row = image_index == len(axes) - 1
        plot_volume(image, axes=image_axes, show=False, cmap=cmap, xlabels=last_row, **plot_volume_kwargs)
        for axis, axis_name in zip(image_axes, axes_names):
            axis.set_title(f'{name} ({axis_name})')
    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path)
    if show:
        plt.show()

def plot_histogram(x: np.ndarray, show=True, **kwargs) -> None:
    _, plt = import_mpl_plt()
    plt.hist(x, bins=get_num_bins(x), **kwargs)
    plt.xlabel('Intensity')
    density = kwargs.pop('density', False)
    ylabel = 'Density' if density else 'Frequency'
    plt.ylabel(ylabel)
    if show:
        plt.show()

def get_num_bins(x: np.ndarray) -> int:
    """Get the optimal number of bins for a histogram.

    This method uses the Freedman–Diaconis rule to compute the histogram that
    minimizes "the integral of the squared difference between the histogram
    (i.e., relative frequency density) and the density of the theoretical
    probability distribution" (`Wikipedia <https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule>`_).

    Args:
        x: Input values.
    """
    q25, q75 = np.percentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
    bins = round((x.max() - x.min()) / bin_width)
    return bins

def color_labels(arrays, cmap_dict):
    results = []
    for slice_array in arrays:
        si, sj, _ = slice_array.shape
        rgb = np.zeros((si, sj, 3), dtype=np.uint8)
        for label, color in cmap_dict.items():
            if isinstance(color, str):
                mpl, _ = import_mpl_plt()
                color = mpl.colors.to_rgb(color)
                color = [255 * n for n in color]
            rgb[slice_array[..., 0] == label] = color
        results.append(rgb)
    return results

def make_gif(tensor: torch.Tensor, axis: int, duration: float, output_path: TypePath, loop: int=0, optimize: bool=True, rescale: bool=True, reverse: bool=False) -> None:
    try:
        from PIL import Image as ImagePIL
    except ModuleNotFoundError as e:
        message = 'Please install Pillow to use Image.to_gif(): pip install Pillow'
        raise RuntimeError(message) from e
    transform = RescaleIntensity((0, 255))
    tensor = transform(tensor) if rescale else tensor
    single_channel = len(tensor) == 1
    axes = np.roll(range(1, 4), -axis)
    tensor = tensor.permute(*axes, 0)
    if single_channel:
        mode = 'P'
        tensor = tensor[..., 0]
    else:
        mode = 'RGB'
    array = tensor.byte().numpy()
    n = 2 if axis == 1 else 1
    images = [ImagePIL.fromarray(rotate(i, n=n)).convert(mode) for i in array]
    num_images = len(images)
    images = list(reversed(images)) if reverse else images
    frame_duration_ms = duration / num_images * 1000
    if frame_duration_ms < 10:
        fps = round(1000 / frame_duration_ms)
        frame_duration_ms = 10
        new_duration = frame_duration_ms * num_images / 1000
        message = f'The computed frame rate from the given duration is too high ({fps} fps). The highest possible frame rate in the GIF file format specification is 100 fps. The duration has been set to {new_duration:.1f} seconds, instead of {duration:.1f}'
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    images[0].save(output_path, save_all=True, append_images=images[1:], optimize=optimize, duration=frame_duration_ms, loop=loop)

class Subject(dict):
    """Class to store information about the images corresponding to a subject.

    Args:
        *args: If provided, a dictionary of items.
        **kwargs: Items that will be added to the subject sample.

    Example:

        >>> import torchio as tio
        >>> # One way:
        >>> subject = tio.Subject(
        ...     one_image=tio.ScalarImage('path_to_image.nii.gz'),
        ...     a_segmentation=tio.LabelMap('path_to_seg.nii.gz'),
        ...     age=45,
        ...     name='John Doe',
        ...     hospital='Hospital Juan Negrín',
        ... )
        >>> # If you want to create the mapping before, or have spaces in the keys:
        >>> subject_dict = {
        ...     'one image': tio.ScalarImage('path_to_image.nii.gz'),
        ...     'a segmentation': tio.LabelMap('path_to_seg.nii.gz'),
        ...     'age': 45,
        ...     'name': 'John Doe',
        ...     'hospital': 'Hospital Juan Negrín',
        ... }
        >>> subject = tio.Subject(subject_dict)
    """

    def __init__(self, *args, **kwargs: dict[str, Any]):
        if args:
            if len(args) == 1 and isinstance(args[0], dict):
                kwargs.update(args[0])
            else:
                message = 'Only one dictionary as positional argument is allowed'
                raise ValueError(message)
        super().__init__(**kwargs)
        self._parse_images(self.get_images(intensity_only=False))
        self.update_attributes()
        self.applied_transforms: list[tuple[str, dict]] = []

    def __repr__(self):
        num_images = len(self.get_images(intensity_only=False))
        string = f'{self.__class__.__name__}(Keys: {tuple(self.keys())}; images: {num_images})'
        return string

    def __len__(self):
        return len(self.get_images(intensity_only=False))

    def __getitem__(self, item):
        if isinstance(item, (slice, int, tuple)):
            try:
                self.check_consistent_spatial_shape()
            except RuntimeError as e:
                message = 'To use indexing, all images in the subject must have the same spatial shape'
                raise RuntimeError(message) from e
            copied = copy.deepcopy(self)
            for image_name, image in copied.items():
                copied[image_name] = image[item]
            return copied
        else:
            return super().__getitem__(item)

    @staticmethod
    def _parse_images(images: list[Image]) -> None:
        if not images:
            raise TypeError('A subject without images cannot be created')

    @property
    def shape(self):
        """Return shape of first image in subject.

        Consistency of shapes across images in the subject is checked first.

        Example:

            >>> import torchio as tio
            >>> colin = tio.datasets.Colin27()
            >>> colin.shape
            (1, 181, 217, 181)
        """
        self.check_consistent_attribute('shape')
        return self.get_first_image().shape

    @property
    def spatial_shape(self):
        """Return spatial shape of first image in subject.

        Consistency of spatial shapes across images in the subject is checked
        first.

        Example:

            >>> import torchio as tio
            >>> colin = tio.datasets.Colin27()
            >>> colin.spatial_shape
            (181, 217, 181)
        """
        self.check_consistent_spatial_shape()
        return self.get_first_image().spatial_shape

    @property
    def spacing(self):
        """Return spacing of first image in subject.

        Consistency of spacings across images in the subject is checked first.

        Example:

            >>> import torchio as tio
            >>> colin = tio.datasets.Slicer()
            >>> colin.spacing
            (1.0, 1.0, 1.2999954223632812)
        """
        self.check_consistent_attribute('spacing')
        return self.get_first_image().spacing

    @property
    def history(self):
        return self.get_applied_transforms()

    def is_2d(self):
        return all((i.is_2d() for i in self.get_images(intensity_only=False)))

    def get_applied_transforms(self, ignore_intensity: bool=False, image_interpolation: str | None=None) -> list[Transform]:
        from ..transforms.intensity_transform import IntensityTransform
        from ..transforms.transform import Transform
        name_to_transform = {cls.__name__: cls for cls in get_subclasses(Transform)}
        transforms_list = []
        for transform_name, arguments in self.applied_transforms:
            transform = name_to_transform[transform_name](**arguments)
            if ignore_intensity and isinstance(transform, IntensityTransform):
                continue
            resamples = hasattr(transform, 'image_interpolation')
            if resamples and image_interpolation is not None:
                parsed = transform.parse_interpolation(image_interpolation)
                transform.image_interpolation = parsed
            transforms_list.append(transform)
        return transforms_list

    def get_composed_history(self, ignore_intensity: bool=False, image_interpolation: str | None=None) -> Compose:
        from ..transforms.augmentation.composition import Compose
        transforms = self.get_applied_transforms(ignore_intensity=ignore_intensity, image_interpolation=image_interpolation)
        return Compose(transforms)

    def get_inverse_transform(self, warn: bool=True, ignore_intensity: bool=False, image_interpolation: str | None=None) -> Compose:
        """Get a reversed list of the inverses of the applied transforms.

        Args:
            warn: Issue a warning if some transforms are not invertible.
            ignore_intensity: If ``True``, all instances of
                :class:`~torchio.transforms.intensity_transform.IntensityTransform`
                will be ignored.
            image_interpolation: Modify interpolation for scalar images inside
                transforms that perform resampling.
        """
        history_transform = self.get_composed_history(ignore_intensity=ignore_intensity, image_interpolation=image_interpolation)
        inverse_transform = history_transform.inverse(warn=warn)
        return inverse_transform

    def apply_inverse_transform(self, **kwargs) -> Subject:
        """Apply the inverse of all applied transforms, in reverse order.

        Args:
            **kwargs: Keyword arguments passed on to
                :meth:`~torchio.data.subject.Subject.get_inverse_transform`.
        """
        inverse_transform = self.get_inverse_transform(**kwargs)
        transformed: Subject
        transformed = inverse_transform(self)
        transformed.clear_history()
        return transformed

    def clear_history(self) -> None:
        self.applied_transforms = []

    def check_consistent_attribute(self, attribute: str, relative_tolerance: float=1e-06, absolute_tolerance: float=1e-06, message: str | None=None) -> None:
        """Check for consistency of an attribute across all images.

        Args:
            attribute: Name of the image attribute to check
            relative_tolerance: Relative tolerance for :func:`numpy.allclose()`
            absolute_tolerance: Absolute tolerance for :func:`numpy.allclose()`

        Example:
            >>> import numpy as np
            >>> import torch
            >>> import torchio as tio
            >>> scalars = torch.randn(1, 512, 512, 100)
            >>> mask = torch.tensor(scalars > 0).type(torch.int16)
            >>> af1 = np.eye([0.8, 0.8, 2.50000000000001, 1])
            >>> af2 = np.eye([0.8, 0.8, 2.49999999999999, 1])  # small difference here (e.g. due to different reader)
            >>> subject = tio.Subject(
            ...   image = tio.ScalarImage(tensor=scalars, affine=af1),
            ...   mask = tio.LabelMap(tensor=mask, affine=af2)
            ... )
            >>> subject.check_consistent_attribute('spacing')  # no error as tolerances are > 0

        .. note:: To check that all values for a specific attribute are close
            between all images in the subject, :func:`numpy.allclose()` is used.
            This function returns ``True`` if
            :math:`|a_i - b_i| \\leq t_{abs} + t_{rel} * |b_i|`, where
            :math:`a_i` and :math:`b_i` are the :math:`i`-th element of the same
            attribute of two images being compared,
            :math:`t_{abs}` is the ``absolute_tolerance`` and
            :math:`t_{rel}` is the ``relative_tolerance``.
        """
        message = f'More than one value for "{attribute}" found in subject images:\n{{}}'
        names_images = self.get_images_dict(intensity_only=False).items()
        try:
            first_attribute = None
            first_image = None
            for image_name, image in names_images:
                if first_attribute is None:
                    first_attribute = getattr(image, attribute)
                    first_image = image_name
                    continue
                current_attribute = getattr(image, attribute)
                all_close = np.allclose(current_attribute, first_attribute, rtol=relative_tolerance, atol=absolute_tolerance)
                if not all_close:
                    message = message.format(pprint.pformat({first_image: first_attribute, image_name: current_attribute}))
                    raise RuntimeError(message)
        except TypeError:
            values_dict = {}
            for image_name, image in names_images:
                values_dict[image_name] = getattr(image, attribute)
            num_unique_values = len(set(values_dict.values()))
            if num_unique_values > 1:
                message = message.format(pprint.pformat(values_dict))
                raise RuntimeError(message) from None

    def check_consistent_spatial_shape(self) -> None:
        self.check_consistent_attribute('spatial_shape')

    def check_consistent_orientation(self) -> None:
        self.check_consistent_attribute('orientation')

    def check_consistent_affine(self) -> None:
        self.check_consistent_attribute('affine')

    def check_consistent_space(self) -> None:
        try:
            self.check_consistent_attribute('spacing')
            self.check_consistent_attribute('direction')
            self.check_consistent_attribute('origin')
            self.check_consistent_spatial_shape()
        except RuntimeError as e:
            message = 'As described above, some images in the subject are not in the same space. You probably can use the transforms ToCanonical and Resample to fix this, as explained at https://github.com/TorchIO-project/torchio/issues/647#issuecomment-913025695'
            raise RuntimeError(message) from e

    def get_images_names(self) -> list[str]:
        return list(self.get_images_dict(intensity_only=False).keys())

    def get_images_dict(self, intensity_only=True, include: Sequence[str] | None=None, exclude: Sequence[str] | None=None) -> dict[str, Image]:
        images = {}
        for image_name, image in self.items():
            if not isinstance(image, Image):
                continue
            if intensity_only and (not image[TYPE] == INTENSITY):
                continue
            if include is not None and image_name not in include:
                continue
            if exclude is not None and image_name in exclude:
                continue
            images[image_name] = image
        return images

    def get_images(self, intensity_only=True, include: Sequence[str] | None=None, exclude: Sequence[str] | None=None) -> list[Image]:
        images_dict = self.get_images_dict(intensity_only=intensity_only, include=include, exclude=exclude)
        return list(images_dict.values())

    def get_image(self, image_name: str) -> Image:
        """Get a single image by its name."""
        return self.get_images_dict(intensity_only=False)[image_name]

    def get_first_image(self) -> Image:
        return self.get_images(intensity_only=False)[0]

    def add_transform(self, transform: Transform, parameters_dict: dict) -> None:
        self.applied_transforms.append((transform.name, parameters_dict))

    def load(self) -> None:
        """Load images in subject on RAM."""
        for image in self.get_images(intensity_only=False):
            image.load()

    def unload(self) -> None:
        """Unload images in subject."""
        for image in self.get_images(intensity_only=False):
            image.unload()

    def update_attributes(self) -> None:
        self.__dict__.update(self)

    @staticmethod
    def _check_image_name(image_name):
        if not isinstance(image_name, str):
            message = f'The image name must be a string, but it has type "{type(image_name)}"'
            raise ValueError(message)
        return image_name

    def add_image(self, image: Image, image_name: str) -> None:
        """Add an image to the subject instance."""
        if not isinstance(image, Image):
            message = f'Image must be an instance of torchio.Image, but its type is "{type(image)}"'
            raise ValueError(message)
        self._check_image_name(image_name)
        self[image_name] = image
        self.update_attributes()

    def remove_image(self, image_name: str) -> None:
        """Remove an image from the subject instance."""
        self._check_image_name(image_name)
        del self[image_name]
        delattr(self, image_name)

    def plot(self, **kwargs) -> None:
        """Plot images using matplotlib.

        Args:
            **kwargs: Keyword arguments that will be passed on to
                :meth:`~torchio.Image.plot`.
        """
        from ..visualization import plot_subject
        plot_subject(self, **kwargs)

class Image(dict):
    """TorchIO image.

    For information about medical image orientation, check out `NiBabel docs`_,
    the `3D Slicer wiki`_, `Graham Wideman's website`_, `FSL docs`_ or
    `SimpleITK docs`_.

    Args:
        path: Path to a file or sequence of paths to files that can be read by
            :mod:`SimpleITK` or :mod:`nibabel`, or to a directory containing
            DICOM files. If :attr:`tensor` is given, the data in
            :attr:`path` will not be read.
            If a sequence of paths is given, data
            will be concatenated on the channel dimension so spatial
            dimensions must match.
        type: Type of image, such as :attr:`torchio.INTENSITY` or
            :attr:`torchio.LABEL`. This will be used by the transforms to
            decide whether to apply an operation, or which interpolation to use
            when resampling. For example, `preprocessing`_ and `augmentation`_
            intensity transforms will only be applied to images with type
            :attr:`torchio.INTENSITY`. Spatial transforms will be applied to
            all types, and nearest neighbor interpolation is always used to
            resample images with type :attr:`torchio.LABEL`.
            The type :attr:`torchio.SAMPLING_MAP` may be used with instances of
            :class:`~torchio.data.sampler.weighted.WeightedSampler`.
        tensor: If :attr:`path` is not given, :attr:`tensor` must be a 4D
            :class:`torch.Tensor` or NumPy array with dimensions
            :math:`(C, W, H, D)`.
        affine: :math:`4 \\times 4` matrix to convert voxel coordinates to world
            coordinates. If ``None``, an identity matrix will be used. See the
            `NiBabel docs on coordinates`_ for more information.
        check_nans: If ``True``, issues a warning if NaNs are found
            in the image. If ``False``, images will not be checked for the
            presence of NaNs.
        reader: Callable object that takes a path and returns a 4D tensor and a
            2D, :math:`4 \\times 4` affine matrix. This can be used if your data
            is saved in a custom format, such as ``.npy`` (see example below).
            If the affine matrix is ``None``, an identity matrix will be used.
        **kwargs: Items that will be added to the image dictionary, e.g.
            acquisition parameters or image ID.
        verify_path: If ``True``, the path will be checked to see if it exists. If
            ``False``, the path will not be verified. This is useful when it is
            expensive to check the path, e.g., when reading a large dataset from a
            mounted drive.

    TorchIO images are `lazy loaders`_, i.e. the data is only loaded from disk
    when needed.

    Example:
        >>> import torchio as tio
        >>> import numpy as np
        >>> image = tio.ScalarImage('t1.nii.gz')  # subclass of Image
        >>> image  # not loaded yet
        ScalarImage(path: t1.nii.gz; type: intensity)
        >>> times_two = 2 * image.data  # data is loaded and cached here
        >>> image
        ScalarImage(shape: (1, 256, 256, 176); spacing: (1.00, 1.00, 1.00); orientation: PIR+; memory: 44.0 MiB; type: intensity)
        >>> image.save('doubled_image.nii.gz')
        >>> def numpy_reader(path):
        ...     data = np.load(path).as_type(np.float32)
        ...     affine = np.eye(4)
        ...     return data, affine
        >>> image = tio.ScalarImage('t1.npy', reader=numpy_reader)

    .. _lazy loaders: https://en.wikipedia.org/wiki/Lazy_loading
    .. _preprocessing: https://docs.torchio.org/transforms/preprocessing.html#intensity
    .. _augmentation: https://docs.torchio.org/transforms/augmentation.html#intensity
    .. _NiBabel docs: https://nipy.org/nibabel/image_orientation.html
    .. _NiBabel docs on coordinates: https://nipy.org/nibabel/coordinate_systems.html#the-affine-matrix-as-a-transformation-between-spaces
    .. _3D Slicer wiki: https://www.slicer.org/wiki/Coordinate_systems
    .. _FSL docs: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Orientation%20Explained
    .. _SimpleITK docs: https://simpleitk.readthedocs.io/en/master/fundamentalConcepts.html
    .. _Graham Wideman's website: http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm
    """

    def __init__(self, path: TypePath | Sequence[TypePath] | None=None, type: str | None=None, tensor: TypeData | None=None, affine: TypeData | None=None, check_nans: bool=False, reader: Callable[[TypePath], TypeDataAffine]=read_image, verify_path: bool=True, **kwargs: dict[str, Any]):
        self.check_nans = check_nans
        self.reader = reader
        if type is None:
            warnings.warn('Not specifying the image type is deprecated and will be mandatory in the future. You can probably use tio.ScalarImage or tio.LabelMap instead', FutureWarning, stacklevel=2)
            type = INTENSITY
        if path is None and tensor is None:
            raise ValueError('A value for path or tensor must be given')
        self._loaded = False
        tensor = self._parse_tensor(tensor)
        affine = self._parse_affine(affine)
        if tensor is not None:
            self.set_data(tensor)
            self.affine = affine
            self._loaded = True
        for key in PROTECTED_KEYS:
            if key in kwargs:
                message = f'Key "{key}" is reserved. Use a different one'
                raise ValueError(message)
        if 'channels_last' in kwargs:
            message = 'The "channels_last" keyword argument is deprecated after https://github.com/TorchIO-project/torchio/pull/685 and will be removed in the future'
            warnings.warn(message, FutureWarning, stacklevel=2)
        super().__init__(**kwargs)
        self.path = self._parse_path(path, verify=verify_path)
        self[PATH] = '' if self.path is None else str(self.path)
        self[STEM] = '' if self.path is None else get_stem(self.path)
        self[TYPE] = type

    def __repr__(self):
        properties = []
        properties.extend([f'shape: {self.shape}', f'spacing: {self.get_spacing_string()}', f'orientation: {self.orientation_str}+'])
        if self._loaded:
            properties.append(f'dtype: {self.data.type()}')
            natural = humanize.naturalsize(self.memory, binary=True)
            properties.append(f'memory: {natural}')
        else:
            properties.append(f'path: "{self.path}"')
        properties = '; '.join(properties)
        string = f'{self.__class__.__name__}({properties})'
        return string

    def __getitem__(self, item):
        if isinstance(item, (slice, int, tuple)):
            return self._crop_from_slices(item)
        if item in (DATA, AFFINE):
            if item not in self:
                self.load()
        return super().__getitem__(item)

    def __array__(self):
        return self.data.numpy()

    def __copy__(self):
        kwargs = {TYPE: self.type, PATH: self.path}
        if self._loaded:
            kwargs[TENSOR] = self.data
            kwargs[AFFINE] = self.affine
        for key, value in self.items():
            if key in PROTECTED_KEYS:
                continue
            kwargs[key] = value
        new_image_class = type(self)
        new_image = new_image_class(check_nans=self.check_nans, reader=self.reader, **kwargs)
        return new_image

    @property
    def data(self) -> torch.Tensor:
        """Tensor data (same as :class:`Image.tensor`)."""
        return self[DATA]

    @data.setter
    @deprecated(version='0.18.16', reason=deprecation_message)
    def data(self, tensor: TypeData):
        self.set_data(tensor)

    def set_data(self, tensor: TypeData):
        """Store a 4D tensor in the :attr:`data` key and attribute.

        Args:
            tensor: 4D tensor with dimensions :math:`(C, W, H, D)`.
        """
        self[DATA] = self._parse_tensor(tensor, none_ok=False)
        self._loaded = True

    @property
    def tensor(self) -> torch.Tensor:
        """Tensor data (same as :class:`Image.data`)."""
        return self.data

    @property
    def affine(self) -> np.ndarray:
        """Affine matrix to transform voxel indices into world coordinates."""
        is_custom_reader = self.reader is not read_image
        if self._loaded or self._is_dir() or self._is_multipath() or is_custom_reader:
            affine = self[AFFINE]
        else:
            assert self.path is not None
            assert isinstance(self.path, (str, Path))
            affine = read_affine(self.path)
        return affine

    @affine.setter
    def affine(self, matrix):
        self[AFFINE] = self._parse_affine(matrix)

    @property
    def type(self) -> str:
        return self[TYPE]

    @property
    def shape(self) -> TypeQuartetInt:
        """Tensor shape as :math:`(C, W, H, D)`."""
        custom_reader = self.reader is not read_image
        multipath = self._is_multipath()
        if isinstance(self.path, Path):
            is_dir = self.path.is_dir()
        shape: TypeQuartetInt
        if self._loaded or custom_reader or multipath or is_dir:
            channels, si, sj, sk = self.data.shape
            shape = (channels, si, sj, sk)
        else:
            assert isinstance(self.path, (str, Path))
            shape = read_shape(self.path)
        return shape

    @property
    def spatial_shape(self) -> TypeTripletInt:
        """Tensor spatial shape as :math:`(W, H, D)`."""
        return self.shape[1:]

    def check_is_2d(self) -> None:
        if not self.is_2d():
            message = f'Image is not 2D. Spatial shape: {self.spatial_shape}'
            raise RuntimeError(message)

    @property
    def height(self) -> int:
        """Image height, if 2D."""
        self.check_is_2d()
        return self.spatial_shape[1]

    @property
    def width(self) -> int:
        """Image width, if 2D."""
        self.check_is_2d()
        return self.spatial_shape[0]

    @property
    def orientation(self) -> tuple[str, str, str]:
        """Orientation codes."""
        return nib.orientations.aff2axcodes(self.affine)

    @property
    def orientation_str(self) -> str:
        """Orientation as a string."""
        return ''.join(self.orientation)

    @property
    def direction(self) -> TypeDirection3D:
        _, _, direction = get_sitk_metadata_from_ras_affine(self.affine, lps=False)
        return direction

    @property
    def spacing(self) -> tuple[float, float, float]:
        """Voxel spacing in mm."""
        _, spacing = get_rotation_and_spacing_from_affine(self.affine)
        sx, sy, sz = spacing
        return (float(sx), float(sy), float(sz))

    @property
    def origin(self) -> tuple[float, float, float]:
        """Center of first voxel in array, in mm."""
        ox, oy, oz = self.affine[:3, 3]
        return (ox, oy, oz)

    @property
    def itemsize(self):
        """Element size of the data type."""
        return self.data.element_size()

    @property
    def memory(self) -> float:
        """Number of Bytes that the tensor takes in the RAM."""
        return np.prod(self.shape) * self.itemsize

    @property
    def bounds(self) -> np.ndarray:
        """Position of centers of voxels in smallest and largest indices."""
        ini = (0, 0, 0)
        fin = np.array(self.spatial_shape) - 1
        point_ini = apply_affine(self.affine, ini)
        point_fin = apply_affine(self.affine, fin)
        return np.array((point_ini, point_fin))

    @property
    def num_channels(self) -> int:
        """Get the number of channels in the associated 4D tensor."""
        return len(self.data)

    def axis_name_to_index(self, axis: str) -> int:
        """Convert an axis name to an axis index.

        Args:
            axis: Possible inputs are ``'Left'``, ``'Right'``, ``'Anterior'``,
                ``'Posterior'``, ``'Inferior'``, ``'Superior'``. Lower-case
                versions and first letters are also valid, as only the first
                letter will be used.

        .. note:: If you are working with animals, you should probably use
            ``'Superior'``, ``'Inferior'``, ``'Anterior'`` and ``'Posterior'``
            for ``'Dorsal'``, ``'Ventral'``, ``'Rostral'`` and ``'Caudal'``,
            respectively.

        .. note:: If your images are 2D, you can use ``'Top'``, ``'Bottom'``,
            ``'Left'`` and ``'Right'``.
        """
        if not isinstance(axis, str):
            raise ValueError('Axis must be a string')
        axis = axis[0].upper()
        if axis in 'TB':
            return -2
        else:
            try:
                index = self.orientation.index(axis)
            except ValueError:
                index = self.orientation.index(self.flip_axis(axis))
            index = -3 + index
            return index

    @staticmethod
    def flip_axis(axis: str) -> str:
        """Return the opposite axis label. For example, ``'L'`` -> ``'R'``.

        Args:
            axis: Axis label, such as ``'L'`` or ``'left'``.
        """
        labels = 'LRPAISTBDV'
        first = labels[::2]
        last = labels[1::2]
        flip_dict = dict(zip(first + last, last + first))
        axis = axis[0].upper()
        flipped_axis = flip_dict.get(axis)
        if flipped_axis is None:
            values = ', '.join(labels)
            message = f'Axis not understood. Please use one of: {values}'
            raise ValueError(message)
        return flipped_axis

    def get_spacing_string(self) -> str:
        strings = [f'{n:.2f}' for n in self.spacing]
        string = f'({', '.join(strings)})'
        return string

    def get_bounds(self) -> TypeBounds:
        """Get minimum and maximum world coordinates occupied by the image."""
        first_index = 3 * (-0.5,)
        last_index = np.array(self.spatial_shape) - 0.5
        first_point = apply_affine(self.affine, first_index)
        last_point = apply_affine(self.affine, last_index)
        array = np.array((first_point, last_point))
        bounds_x, bounds_y, bounds_z = array.T.tolist()
        return (bounds_x, bounds_y, bounds_z)

    def _parse_single_path(self, path: TypePath, *, verify: bool=True) -> Path:
        if isinstance(path, (torch.Tensor, np.ndarray)):
            class_name = self.__class__.__name__
            message = f'Expected type str or Path but found a tensor/array. Instead of {class_name}(your_tensor), use {class_name}(tensor=your_tensor).'
            raise TypeError(message)
        try:
            path = Path(path).expanduser()
        except TypeError as err:
            message = f'Expected type str or Path but found an object with type {type(path)} instead'
            raise TypeError(message) from err
        except RuntimeError as err:
            message = f'Conversion to path not possible for variable: {path}'
            raise RuntimeError(message) from err
        if not verify:
            return path
        if not (path.is_file() or path.is_dir()):
            raise FileNotFoundError(f'File not found: "{path}"')
        return path

    def _parse_path(self, path: TypePath | Sequence[TypePath] | None, *, verify: bool=True) -> Path | list[Path] | None:
        if path is None:
            return None
        elif isinstance(path, dict):
            raise TypeError('The path argument cannot be a dictionary')
        elif self._is_paths_sequence(path):
            return [self._parse_single_path(p, verify=verify) for p in path]
        else:
            return self._parse_single_path(path, verify=verify)

    def _parse_tensor(self, tensor: TypeData | None, none_ok: bool=True) -> torch.Tensor | None:
        if tensor is None:
            if none_ok:
                return None
            else:
                raise RuntimeError('Input tensor cannot be None')
        if isinstance(tensor, np.ndarray):
            tensor = check_uint_to_int(tensor)
            tensor = torch.as_tensor(tensor)
        elif not isinstance(tensor, torch.Tensor):
            message = f'Input tensor must be a PyTorch tensor or NumPy array, but type "{type(tensor)}" was found'
            raise TypeError(message)
        ndim = tensor.ndim
        if ndim != 4:
            raise ValueError(f'Input tensor must be 4D, but it is {ndim}D')
        if tensor.dtype == torch.bool:
            tensor = tensor.to(torch.uint8)
        if self.check_nans and torch.isnan(tensor).any():
            warnings.warn('NaNs found in tensor', RuntimeWarning, stacklevel=2)
        return tensor

    @staticmethod
    def _parse_tensor_shape(tensor: torch.Tensor) -> TypeData:
        return ensure_4d(tensor)

    @staticmethod
    def _parse_affine(affine: TypeData | None) -> np.ndarray:
        if affine is None:
            return np.eye(4)
        if isinstance(affine, torch.Tensor):
            affine = affine.numpy()
        if not isinstance(affine, np.ndarray):
            bad_type = type(affine)
            raise TypeError(f'Affine must be a NumPy array, not {bad_type}')
        if affine.shape != (4, 4):
            bad_shape = affine.shape
            raise ValueError(f'Affine shape must be (4, 4), not {bad_shape}')
        return affine.astype(np.float64)

    @staticmethod
    def _is_paths_sequence(path: TypePath | Sequence[TypePath] | None) -> bool:
        is_not_string = not isinstance(path, str)
        return is_not_string and is_iterable(path)

    def _is_multipath(self) -> bool:
        return self._is_paths_sequence(self.path)

    def _is_dir(self) -> bool:
        is_sequence = self._is_multipath()
        if is_sequence:
            return False
        elif self.path is None:
            return False
        else:
            assert isinstance(self.path, Path)
            return self.path.is_dir()

    def load(self) -> None:
        """Load the image from disk.

        Returns:
            Tuple containing a 4D tensor of size :math:`(C, W, H, D)` and a 2D
            :math:`4 \\times 4` affine matrix to convert voxel indices to world
            coordinates.
        """
        if self._loaded:
            return
        paths: list[Path]
        if self._is_multipath():
            paths = self.path
        else:
            paths = [self.path]
        tensor, affine = self.read_and_check(paths[0])
        tensors = [tensor]
        for path in paths[1:]:
            new_tensor, new_affine = self.read_and_check(path)
            if not np.array_equal(affine, new_affine):
                message = f'Files have different affine matrices.\nMatrix of {paths[0]}:\n{affine}\nMatrix of {path}:\n{new_affine}'
                warnings.warn(message, RuntimeWarning, stacklevel=2)
            if not tensor.shape[1:] == new_tensor.shape[1:]:
                message = f'Files shape do not match, found {tensor.shape}and {new_tensor.shape}'
                raise RuntimeError(message)
            tensors.append(new_tensor)
        tensor = torch.cat(tensors)
        self.set_data(tensor)
        self.affine = affine
        self._loaded = True

    def unload(self) -> None:
        """Unload the image from memory.

        Raises:
            RuntimeError: If the images has not been loaded yet or if no path
                is available.
        """
        if not self._loaded:
            message = 'Image cannot be unloaded as it has not been loaded yet'
            raise RuntimeError(message)
        if self.path is None:
            message = 'Cannot unload image as no path is available from where the image could be loaded again'
            raise RuntimeError(message)
        self[DATA] = None
        self[AFFINE] = None
        self._loaded = False

    def read_and_check(self, path: TypePath) -> TypeDataAffine:
        tensor, affine = self.reader(path)
        if self.reader is not read_image and isinstance(tensor, np.ndarray):
            tensor = check_uint_to_int(tensor)
        tensor = self._parse_tensor_shape(tensor)
        tensor = self._parse_tensor(tensor)
        affine = self._parse_affine(affine)
        if self.check_nans and torch.isnan(tensor).any():
            warnings.warn(f'NaNs found in file "{path}"', RuntimeWarning, stacklevel=2)
        return (tensor, affine)

    def save(self, path: TypePath, squeeze: bool | None=None) -> None:
        """Save image to disk.

        Args:
            path: String or instance of :class:`pathlib.Path`.
            squeeze: Whether to remove singleton dimensions before saving.
                If ``None``, the array will be squeezed if the output format is
                JP(E)G, PNG, BMP or TIF(F).
        """
        write_image(self.data, self.affine, path, squeeze=squeeze)

    def is_2d(self) -> bool:
        return self.shape[-1] == 1

    def numpy(self) -> np.ndarray:
        """Get a NumPy array containing the image data."""
        return np.asarray(self)

    def as_sitk(self, **kwargs) -> sitk.Image:
        """Get the image as an instance of :class:`sitk.Image`."""
        return nib_to_sitk(self.data, self.affine, **kwargs)

    @classmethod
    def from_sitk(cls, sitk_image):
        """Instantiate a new TorchIO image from a :class:`sitk.Image`.

        Example:
            >>> import torchio as tio
            >>> import SimpleITK as sitk
            >>> sitk_image = sitk.Image(20, 30, 40, sitk.sitkUInt16)
            >>> tio.LabelMap.from_sitk(sitk_image)
            LabelMap(shape: (1, 20, 30, 40); spacing: (1.00, 1.00, 1.00); orientation: LPS+; memory: 93.8 KiB; dtype: torch.IntTensor)
            >>> sitk_image = sitk.Image((224, 224), sitk.sitkVectorFloat32, 3)
            >>> tio.ScalarImage.from_sitk(sitk_image)
            ScalarImage(shape: (3, 224, 224, 1); spacing: (1.00, 1.00, 1.00); orientation: LPS+; memory: 588.0 KiB; dtype: torch.FloatTensor)
        """
        tensor, affine = sitk_to_nib(sitk_image)
        return cls(tensor=tensor, affine=affine)

    def as_pil(self, transpose=True):
        """Get the image as an instance of :class:`PIL.Image`.

        .. note:: Values will be clamped to 0-255 and cast to uint8.

        .. note:: To use this method, Pillow needs to be installed:
            ``pip install Pillow``.
        """
        try:
            from PIL import Image as ImagePIL
        except ModuleNotFoundError as e:
            message = 'Please install Pillow to use Image.as_pil(): pip install Pillow'
            raise RuntimeError(message) from e
        self.check_is_2d()
        tensor = self.data
        if len(tensor) not in (1, 3, 4):
            raise NotImplementedError('Only 1, 3 or 4 channels are supported for conversion to Pillow image')
        if len(tensor) == 1:
            tensor = torch.cat(3 * [tensor])
        if transpose:
            tensor = tensor.permute(3, 2, 1, 0)
        else:
            tensor = tensor.permute(3, 1, 2, 0)
        array = tensor.clamp(0, 255).numpy()[0]
        return ImagePIL.fromarray(array.astype(np.uint8))

    def to_gif(self, axis: int, duration: float, output_path: TypePath, loop: int=0, rescale: bool=True, optimize: bool=True, reverse: bool=False) -> None:
        """Save an animated GIF of the image.

        Args:
            axis: Spatial axis (0, 1 or 2).
            duration: Duration of the full animation in seconds.
            output_path: Path to the output GIF file.
            loop: Number of times the GIF should loop.
                ``0`` means that it will loop forever.
            rescale: Use :class:`~torchio.transforms.preprocessing.intensity.rescale.RescaleIntensity`
                to rescale the intensity values to :math:`[0, 255]`.
            optimize: If ``True``, attempt to compress the palette by
                eliminating unused colors. This is only useful if the palette
                can be compressed to the next smaller power of 2 elements.
            reverse: Reverse the temporal order of frames.
        """
        from ..visualization import make_gif
        make_gif(self.data, axis, duration, output_path, loop=loop, rescale=rescale, optimize=optimize, reverse=reverse)

    def to_ras(self) -> Image:
        if self.orientation_str != 'RAS':
            from ..transforms.preprocessing.spatial.to_canonical import ToCanonical
            return ToCanonical()(self)
        return self

    def get_center(self, lps: bool=False) -> TypeTripletFloat:
        """Get image center in RAS+ or LPS+ coordinates.

        Args:
            lps: If ``True``, the coordinates will be in LPS+ orientation, i.e.
                the first dimension grows towards the left, etc. Otherwise, the
                coordinates will be in RAS+ orientation.
        """
        size = np.array(self.spatial_shape)
        center_index = (size - 1) / 2
        r, a, s = apply_affine(self.affine, center_index)
        if lps:
            return (-r, -a, s)
        else:
            return (r, a, s)

    def set_check_nans(self, check_nans: bool) -> None:
        self.check_nans = check_nans

    def plot(self, **kwargs) -> None:
        """Plot image."""
        if self.is_2d():
            self.as_pil().show()
        else:
            from ..visualization import plot_volume
            plot_volume(self, **kwargs)

    def show(self, viewer_path: TypePath | None=None) -> None:
        """Open the image using external software.

        Args:
            viewer_path: Path to the application used to view the image. If
                ``None``, the value of the environment variable
                ``SITK_SHOW_COMMAND`` will be used. If this variable is also
                not set, TorchIO will try to guess the location of
                `ITK-SNAP <http://www.itksnap.org/pmwiki/pmwiki.php>`_ and
                `3D Slicer <https://www.slicer.org/>`_.

        Raises:
            RuntimeError: If the viewer is not found.
        """
        sitk_image = self.as_sitk()
        image_viewer = sitk.ImageViewer()
        if self.__class__.__name__ == 'LabelMap':
            image_viewer.SetFileExtension('.seg.nrrd')
        if viewer_path is not None:
            image_viewer.SetApplication(str(viewer_path))
        try:
            image_viewer.Execute(sitk_image)
        except RuntimeError as e:
            viewer_path = guess_external_viewer()
            if viewer_path is None:
                message = 'No external viewer has been found. Please set the environment variable SITK_SHOW_COMMAND to a viewer of your choice'
                raise RuntimeError(message) from e
            image_viewer.SetApplication(str(viewer_path))
            image_viewer.Execute(sitk_image)

    def _crop_from_slices(self, slices: TypeSlice | tuple[TypeSlice, ...]) -> Image:
        from ..transforms import Crop
        slices_tuple = to_tuple(slices)
        cropping: list[int] = []
        for dim, slice_ in enumerate(slices_tuple):
            if isinstance(slice_, slice):
                pass
            elif slice_ is Ellipsis:
                message = 'Ellipsis slicing is not supported yet'
                raise NotImplementedError(message)
            elif isinstance(slice_, int):
                slice_ = slice(slice_, slice_ + 1)
            else:
                message = f'Slice type not understood: "{type(slice_)}"'
                raise TypeError(message)
            shape_dim = self.spatial_shape[dim]
            assert isinstance(slice_, slice)
            start, stop, step = slice_.indices(shape_dim)
            if step != 1:
                message = 'Slicing with steps different from 1 is not supported yet. Use the Crop transform instead'
                raise ValueError(message)
            crop_ini = start
            crop_fin = shape_dim - stop
            cropping.extend([crop_ini, crop_fin])
        while dim < 2:
            cropping.extend([0, 0])
            dim += 1
        w_ini, w_fin, h_ini, h_fin, d_ini, d_fin = cropping
        cropping_arg = (w_ini, w_fin, h_ini, h_fin, d_ini, d_fin)
        return Crop(cropping_arg)(self)

class ScalarImage(Image):
    """Image whose pixel values represent scalars.

    Example:
        >>> import torch
        >>> import torchio as tio
        >>> # Loading from a file
        >>> t1_image = tio.ScalarImage('t1.nii.gz')
        >>> dmri = tio.ScalarImage(tensor=torch.rand(32, 128, 128, 88))
        >>> image = tio.ScalarImage('safe_image.nrrd', check_nans=False)
        >>> data, affine = image.data, image.affine
        >>> affine.shape
        (4, 4)
        >>> image.data is image[tio.DATA]
        True
        >>> image.data is image.tensor
        True
        >>> type(image.data)
        torch.Tensor

    See :class:`~torchio.Image` for more information.
    """

    def __init__(self, *args, **kwargs):
        if 'type' in kwargs and kwargs['type'] != INTENSITY:
            raise ValueError('Type of ScalarImage is always torchio.INTENSITY')
        kwargs.update({'type': INTENSITY})
        super().__init__(*args, **kwargs)

    def hist(self, **kwargs) -> None:
        """Plot histogram."""
        from ..visualization import plot_histogram
        x = self.data.flatten().numpy()
        plot_histogram(x, **kwargs)

    def to_video(self, output_path: TypePath, frame_rate: float | None=15, seconds: float | None=None, direction: str='I', verbosity: str='error') -> None:
        """Create a video showing all image slices along a specified direction.

        Args:
            output_path: Path to the output video file.
            frame_rate: Number of frames per second (FPS).
            seconds: Target duration of the full video.
            direction:
            verbosity:

        .. note:: Only ``frame_rate`` or ``seconds`` may (and must) be specified.
        """
        from ..visualization import make_video
        make_video(self.to_ras(), output_path, frame_rate=frame_rate, seconds=seconds, direction=direction, verbosity=verbosity)

