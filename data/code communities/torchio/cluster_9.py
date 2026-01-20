# Cluster 9

def parse_spatial_shape(shape):
    result = to_tuple(shape, length=3)
    for n in result:
        if n < 1 or n % 1:
            message = f'All elements in a spatial shape must be positive integers, but the following shape was passed: {shape}'
            raise ValueError(message)
    if len(result) != 3:
        message = f'Spatial shapes must have 3 elements, but the following shape was passed: {shape}'
        raise ValueError(message)
    return result

def to_tuple(value: Any, length: int=1) -> tuple[TypeNumber, ...]:
    """Convert variable to tuple of length n.

    Example:
        >>> from torchio.utils import to_tuple
        >>> to_tuple(1, length=1)
        (1,)
        >>> to_tuple(1, length=3)
        (1, 1, 1)

    If value is an iterable, n is ignored and tuple(value) is returned

    Example:
        >>> to_tuple((1,), length=1)
        (1,)
        >>> to_tuple((1, 2), length=1)
        (1, 2)
        >>> to_tuple([1, 2], length=3)
        (1, 2)
    """
    try:
        iter(value)
        value = tuple(value)
    except TypeError:
        value = length * (value,)
    return value

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

class GridSampler(PatchSampler):
    """Extract patches across a whole volume.

    Grid samplers are useful to perform inference using all patches from a
    volume. It is often used with a :class:`~torchio.data.GridAggregator`.

    Args:
        subject: Instance of :class:`~torchio.data.Subject`
            from which patches will be extracted.
        patch_size: Tuple of integers :math:`(w, h, d)` to generate patches
            of size :math:`w \\times h \\times d`.
            If a single number :math:`n` is provided,
            :math:`w = h = d = n`.
        patch_overlap: Tuple of even integers :math:`(w_o, h_o, d_o)`
            specifying the overlap between patches for dense inference. If a
            single number :math:`n` is provided, :math:`w_o = h_o = d_o = n`.
        padding_mode: Same as :attr:`padding_mode` in
            :class:`~torchio.transforms.Pad`. If ``None``, the volume will not
            be padded before sampling and patches at the border will not be
            cropped by the aggregator.
            Otherwise, the volume will be padded with
            :math:`\\left(\\frac{w_o}{2}, \\frac{h_o}{2}, \\frac{d_o}{2} \\right)`
            on each side before sampling. If the sampler is passed to a
            :class:`~torchio.data.GridAggregator`, it will crop the output
            to its original size.

    Example:

        >>> import torchio as tio
        >>> colin = tio.datasets.Colin27()
        >>> sampler = tio.GridSampler(colin, patch_size=88)
        >>> for i, patch in enumerate(sampler()):
        ...     patch.t1.save(f'patch_{i}.nii.gz')
        ...
        >>> # To figure out the number of patches beforehand:
        >>> sampler = tio.GridSampler(colin, patch_size=88)
        >>> len(sampler)
        8

    .. note:: Adapted from NiftyNet. See `this NiftyNet tutorial
        <https://niftynet.readthedocs.io/en/dev/window_sizes.html>`_ for more
        information about patch based sampling. Note that
        :attr:`patch_overlap` is twice :attr:`border` in NiftyNet
        tutorial.
    """

    def __init__(self, subject: Subject, patch_size: TypeSpatialShape, patch_overlap: TypeSpatialShape=(0, 0, 0), padding_mode: str | float | None=None):
        super().__init__(patch_size)
        self.patch_overlap = np.array(to_tuple(patch_overlap, length=3))
        self.padding_mode = padding_mode
        self.subject = self._pad(subject)
        self.locations = self._compute_locations(self.subject)

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, index):
        location = self.locations[index]
        index_ini = location[:3]
        cropped_subject = self.crop(self.subject, index_ini, self.patch_size)
        return cropped_subject

    def __call__(self, subject: Subject | None=None, num_patches: int | None=None) -> Generator[Subject]:
        subject = self.subject if subject is None else subject
        return super().__call__(subject, num_patches=num_patches)

    def _pad(self, subject: Subject) -> Subject:
        if self.padding_mode is not None:
            from ...transforms import Pad
            border = self.patch_overlap // 2
            padding = border.repeat(2)
            pad = Pad(padding, padding_mode=self.padding_mode)
            subject = pad(subject)
        return subject

    def _compute_locations(self, subject: Subject):
        sizes = (subject.spatial_shape, self.patch_size, self.patch_overlap)
        self._parse_sizes(*sizes)
        return self._get_patches_locations(*sizes)

    def _generate_patches(self, subject: Subject) -> Generator[Subject]:
        subject = self._pad(subject)
        sizes = (subject.spatial_shape, self.patch_size, self.patch_overlap)
        self._parse_sizes(*sizes)
        locations = self._get_patches_locations(*sizes)
        for location in locations:
            index_ini = location[:3]
            yield self.extract_patch(subject, index_ini)

    @staticmethod
    def _parse_sizes(image_size: TypeTripletInt, patch_size: TypeTripletInt, patch_overlap: TypeTripletInt) -> None:
        image_size_array = np.array(image_size)
        patch_size_array = np.array(patch_size)
        patch_overlap_array = np.array(patch_overlap)
        if np.any(patch_size_array > image_size_array):
            message = f'Patch size {tuple(patch_size_array)} cannot be larger than image size {tuple(image_size_array)}'
            raise ValueError(message)
        if np.any(patch_overlap_array >= patch_size_array):
            message = f'Patch overlap {tuple(patch_overlap_array)} must be smaller than patch size {tuple(patch_size_array)}'
            raise ValueError(message)
        if np.any(patch_overlap_array % 2):
            message = f'Patch overlap must be a tuple of even integers, not {tuple(patch_overlap_array)}'
            raise ValueError(message)

    @staticmethod
    def _get_patches_locations(image_size: TypeTripletInt, patch_size: TypeTripletInt, patch_overlap: TypeTripletInt) -> np.ndarray:
        indices = []
        zipped = zip(image_size, patch_size, patch_overlap)
        for im_size_dim, patch_size_dim, patch_overlap_dim in zipped:
            end = im_size_dim + 1 - patch_size_dim
            step = patch_size_dim - patch_overlap_dim
            indices_dim = list(range(0, end, step))
            if indices_dim[-1] != im_size_dim - patch_size_dim:
                indices_dim.append(im_size_dim - patch_size_dim)
            indices.append(indices_dim)
        indices_ini = np.array(np.meshgrid(*indices)).reshape(3, -1).T
        indices_ini = np.unique(indices_ini, axis=0)
        indices_fin = indices_ini + np.array(patch_size)
        locations = np.hstack((indices_ini, indices_fin))
        return np.array(sorted(locations.tolist()))

class PatchSampler:
    """Base class for TorchIO samplers.

    Args:
        patch_size: Tuple of integers :math:`(w, h, d)` to generate patches
            of size :math:`w \\times h \\times d`.
            If a single number :math:`n` is provided, :math:`w = h = d = n`.

    .. warning:: This is an abstract class that should only be instantiated
        using child classes such as :class:`~torchio.data.UniformSampler` and
        :class:`~torchio.data.WeightedSampler`.
    """

    def __init__(self, patch_size: TypeSpatialShape):
        patch_size_array = np.array(to_tuple(patch_size, length=3))
        for n in patch_size_array:
            if n < 1 or not isinstance(n, (int, np.integer)):
                message = f'Patch dimensions must be positive integers, not {patch_size_array}'
                raise ValueError(message)
        self.patch_size = patch_size_array.astype(np.uint16)

    def extract_patch(self, subject: Subject, index_ini: TypeTripletInt) -> Subject:
        cropped_subject = self.crop(subject, index_ini, self.patch_size)
        return cropped_subject

    def crop(self, subject: Subject, index_ini: TypeTripletInt, patch_size: TypeTripletInt) -> Subject:
        transform = self._get_crop_transform(subject, index_ini, patch_size)
        cropped_subject = transform(subject)
        index_ini_array = np.asarray(index_ini)
        patch_size_array = np.asarray(patch_size)
        index_fin = index_ini_array + patch_size_array
        location = index_ini_array.tolist() + index_fin.tolist()
        cropped_subject[LOCATION] = torch.as_tensor(location)
        cropped_subject.update_attributes()
        return cropped_subject

    @staticmethod
    def _get_crop_transform(subject, index_ini: TypeTripletInt, patch_size: TypeSpatialShape):
        from ...transforms.preprocessing.spatial.crop import Crop
        shape = np.array(subject.spatial_shape, dtype=np.uint16)
        index_ini_array = np.array(index_ini, dtype=np.uint16)
        patch_size_array = np.array(patch_size, dtype=np.uint16)
        assert len(index_ini_array) == 3
        assert len(patch_size_array) == 3
        index_fin = index_ini_array + patch_size_array
        crop_ini = index_ini_array.tolist()
        crop_fin = (shape - index_fin).tolist()
        start = ()
        cropping = sum(zip(crop_ini, crop_fin), start)
        return Crop(cropping)

    def __call__(self, subject: Subject, num_patches: int | None=None) -> Generator[Subject]:
        subject.check_consistent_space()
        if np.any(self.patch_size > subject.spatial_shape):
            message = f'Patch size {tuple(self.patch_size)} cannot be larger than image size {tuple(subject.spatial_shape)}'
            raise RuntimeError(message)
        kwargs = {} if num_patches is None else {'num_patches': num_patches}
        return self._generate_patches(subject, **kwargs)

    def _generate_patches(self, subject: Subject, num_patches: int | None=None) -> Generator[Subject]:
        raise NotImplementedError

class Transform(ABC):
    """Abstract class for all TorchIO transforms.

    When called, the input can be an instance of
    :class:`torchio.Subject`,
    :class:`torchio.Image`,
    :class:`numpy.ndarray`,
    :class:`torch.Tensor`,
    :class:`SimpleITK.Image`,
    or :class:`dict` containing 4D tensors as values.

    All subclasses must overwrite
    :meth:`~torchio.transforms.Transform.apply_transform`,
    which takes an instance of :class:`~torchio.Subject`,
    modifies it and returns the result.

    Args:
        p: Probability that this transform will be applied.
        copy: Make a deep copy of the input before applying the transform.
        include: Sequence of strings with the names of the only images to which
            the transform will be applied.
            Mandatory if the input is a :class:`dict`.
        exclude: Sequence of strings with the names of the images to which the
            the transform will not be applied, apart from the ones that are
            excluded because of the transform type.
            For example, if a subject includes an MRI, a CT and a label map,
            and the CT is added to the list of exclusions of an intensity
            transform such as :class:`~torchio.transforms.RandomBlur`,
            the transform will be only applied to the MRI, as the label map is
            excluded by default by spatial transforms.
        keep: Dictionary with the names of the input images that will be kept
            in the output and their new names. For example:
            ``{'t1': 't1_original'}``. This might be useful for autoencoders
            or registration tasks.
        parse_input: If ``True``, the input will be converted to an instance of
            :class:`~torchio.Subject`. This is used internally by some special
            transforms like
            :class:`~torchio.transforms.augmentation.composition.Compose`.
        label_keys: If the input is a dictionary, names of images that
            correspond to label maps.
    """

    def __init__(self, p: float=1, copy: bool=True, include: TypeKeys=None, exclude: TypeKeys=None, keys: TypeKeys=None, keep: dict[str, str] | None=None, parse_input: bool=True, label_keys: TypeKeys=None):
        self.probability = self.parse_probability(p)
        self.copy = copy
        if keys is not None:
            message = 'The "keys" argument is deprecated and will be removed in the future. Use "include" instead'
            warnings.warn(message, FutureWarning, stacklevel=2)
            include = keys
        self.include, self.exclude = self.parse_include_and_exclude_keys(include, exclude, label_keys)
        self.keep = keep
        self.parse_input = parse_input
        self.label_keys = label_keys
        self.args_names: list[str] = []

    def __call__(self, data: InputType) -> InputType:
        """Transform data and return a result of the same type.

        Args:
            data: Instance of :class:`torchio.Subject`, 4D
                :class:`torch.Tensor` or :class:`numpy.ndarray` with dimensions
                :math:`(C, W, H, D)`, where :math:`C` is the number of channels
                and :math:`W, H, D` are the spatial dimensions. If the input is
                a tensor, the affine matrix will be set to identity. Other
                valid input types are a SimpleITK image, a
                :class:`torchio.Image`, a NiBabel Nifti1 image or a
                :class:`dict`. The output type is the same as the input type.
        """
        if torch.rand(1).item() > self.probability:
            return data
        if self.parse_input:
            data_parser = DataParser(data, keys=self.include, label_keys=self.label_keys)
            subject = data_parser.get_subject()
        else:
            subject = data
        if self.keep is not None:
            images_to_keep = {}
            for name, new_name in self.keep.items():
                images_to_keep[new_name] = copy.deepcopy(subject[name])
        if self.copy:
            subject = copy.deepcopy(subject)
        with np.errstate(all='raise', under='ignore'):
            transformed = self.apply_transform(subject)
        if self.keep is not None:
            for name, image in images_to_keep.items():
                transformed.add_image(image, name)
        if self.parse_input:
            self.add_transform_to_subject_history(transformed)
            for image in transformed.get_images(intensity_only=False):
                ndim = image.data.ndim
                assert ndim == 4, f'Output of {self.name} is {ndim}D'
            output = data_parser.get_output(transformed)
        else:
            output = transformed
        return output

    def __repr__(self):
        if hasattr(self, 'args_names'):
            names = self.args_names
            args_strings = [f'{arg}={getattr(self, arg)}' for arg in names]
            if hasattr(self, 'invert_transform') and self.invert_transform:
                args_strings.append('invert=True')
            args_string = ', '.join(args_strings)
            return f'{self.name}({args_string})'
        else:
            return super().__repr__()

    def get_base_args(self) -> dict:
        """Provides easy access to the arguments used to instantiate the base class
        (:class:`~torchio.transforms.transform.Transform`) of any transform.

        This method is particularly useful when a new transform can be represented as a variant
        of an existing transform (e.g. all random transforms), allowing for seamless instantiation
        of the existing transform with the same arguments as the new transform during `apply_transform`.

        Note: The `p` argument (probability of applying the transform) is excluded to avoid
        multiplying the probability of both existing and new transform.
        """
        return {'copy': self.copy, 'include': self.include, 'exclude': self.exclude, 'keep': self.keep, 'parse_input': self.parse_input, 'label_keys': self.label_keys}

    def add_base_args(self, arguments, overwrite_on_existing: bool=False):
        """Add the init args to existing arguments"""
        for key, value in self.get_base_args().items():
            if key in arguments and (not overwrite_on_existing):
                continue
            arguments[key] = value
        return arguments

    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def apply_transform(self, subject: Subject) -> Subject:
        raise NotImplementedError

    def add_transform_to_subject_history(self, subject):
        from . import Compose
        from . import CropOrPad
        from . import EnsureShapeMultiple
        from . import OneOf
        from .augmentation import RandomTransform
        from .preprocessing import Resize
        from .preprocessing import SequentialLabels
        call_others = (RandomTransform, Compose, OneOf, CropOrPad, EnsureShapeMultiple, SequentialLabels, Resize)
        if not isinstance(self, call_others):
            subject.add_transform(self, self._get_reproducing_arguments())

    @staticmethod
    def to_range(n, around):
        if around is None:
            return (0, n)
        else:
            return (around - n, around + n)

    def parse_params(self, params, around, name, make_ranges=True, **kwargs):
        params = to_tuple(params)
        if len(params) == 1 or (len(params) == 2 and make_ranges):
            params *= 3
        if len(params) == 3 and make_ranges:
            items = [self.to_range(n, around) for n in params]
            params = [n for prange in items for n in prange]
        if make_ranges:
            if len(params) != 6:
                message = f'If "{name}" is a sequence, it must have length 2, 3 or 6, not {len(params)}'
                raise ValueError(message)
            for param_range in zip(params[::2], params[1::2]):
                self._parse_range(param_range, name, **kwargs)
        return tuple(params)

    @staticmethod
    def _parse_range(nums_range: TypeNumber | tuple[TypeNumber, TypeNumber], name: str, min_constraint: TypeNumber | None=None, max_constraint: TypeNumber | None=None, type_constraint: type | None=None) -> tuple[TypeNumber, TypeNumber]:
        """Adapted from :class:`torchvision.transforms.RandomRotation`.

        Args:
            nums_range: Tuple of two numbers :math:`(n_{min}, n_{max})`,
                where :math:`n_{min} \\leq n_{max}`.
                If a single positive number :math:`n` is provided,
                :math:`n_{min} = -n` and :math:`n_{max} = n`.
            name: Name of the parameter, so that an informative error message
                can be printed.
            min_constraint: Minimal value that :math:`n_{min}` can take,
                default is None, i.e. there is no minimal value.
            max_constraint: Maximal value that :math:`n_{max}` can take,
                default is None, i.e. there is no maximal value.
            type_constraint: Precise type that :math:`n_{max}` and
                :math:`n_{min}` must take.

        Returns:
            A tuple of two numbers :math:`(n_{min}, n_{max})`.

        Raises:
            ValueError: if :attr:`nums_range` is negative
            ValueError: if :math:`n_{max}` or :math:`n_{min}` is not a number
            ValueError: if :math:`n_{max} \\lt n_{min}`
            ValueError: if :attr:`min_constraint` is not None and
                :math:`n_{min}` is smaller than :attr:`min_constraint`
            ValueError: if :attr:`max_constraint` is not None and
                :math:`n_{max}` is greater than :attr:`max_constraint`
            ValueError: if :attr:`type_constraint` is not None and
                :math:`n_{max}` and :math:`n_{max}` are not of type
                :attr:`type_constraint`.
        """
        if isinstance(nums_range, numbers.Number):
            if nums_range < 0:
                raise ValueError(f'If {name} is a single number, it must be positive, not {nums_range}')
            if min_constraint is not None and nums_range < min_constraint:
                raise ValueError(f'If {name} is a single number, it must be greater than {min_constraint}, not {nums_range}')
            if max_constraint is not None and nums_range > max_constraint:
                raise ValueError(f'If {name} is a single number, it must be smaller than {max_constraint}, not {nums_range}')
            if type_constraint is not None:
                if not isinstance(nums_range, type_constraint):
                    raise ValueError(f'If {name} is a single number, it must be of type {type_constraint}, not {nums_range}')
            min_range = -nums_range if min_constraint is None else nums_range
            return (min_range, nums_range)
        try:
            min_value, max_value = nums_range
        except (TypeError, ValueError) as err:
            message = f'If {name} is not a single number, it must be a sequence of len 2, not {nums_range}'
            raise ValueError(message) from err
        min_is_number = isinstance(min_value, numbers.Number)
        max_is_number = isinstance(max_value, numbers.Number)
        if not min_is_number or not max_is_number:
            message = f'{name} values must be numbers, not {nums_range}'
            raise ValueError(message)
        if min_value > max_value:
            raise ValueError(f'If {name} is a sequence, the second value must be equal or greater than the first, but it is {nums_range}')
        if min_constraint is not None and min_value < min_constraint:
            raise ValueError(f'If {name} is a sequence, the first value must be greater than {min_constraint}, but it is {min_value}')
        if max_constraint is not None and max_value > max_constraint:
            raise ValueError(f'If {name} is a sequence, the second value must be smaller than {max_constraint}, but it is {max_value}')
        if type_constraint is not None:
            min_type_ok = isinstance(min_value, type_constraint)
            max_type_ok = isinstance(max_value, type_constraint)
            if not min_type_ok or not max_type_ok:
                raise ValueError(f'If "{name}" is a sequence, its values must be of type "{type_constraint}", not "{type(nums_range)}"')
        return nums_range

    @staticmethod
    def parse_interpolation(interpolation: str) -> str:
        if not isinstance(interpolation, str):
            itype = type(interpolation)
            raise TypeError(f'Interpolation must be a string, not {itype}')
        interpolation = interpolation.lower()
        is_string = isinstance(interpolation, str)
        supported_values = [key.name.lower() for key in Interpolation]
        is_supported = interpolation.lower() in supported_values
        if is_string and is_supported:
            return interpolation
        message = f'Interpolation "{interpolation}" of type {type(interpolation)} must be a string among the supported values: {supported_values}'
        raise ValueError(message)

    @staticmethod
    def parse_probability(probability: float) -> float:
        is_number = isinstance(probability, numbers.Number)
        if not (is_number and 0 <= probability <= 1):
            message = f'Probability must be a number in [0, 1], not {probability}'
            raise ValueError(message)
        return probability

    @staticmethod
    def parse_include_and_exclude_keys(include: TypeKeys, exclude: TypeKeys, label_keys: TypeKeys) -> tuple[TypeKeys, TypeKeys]:
        if include is not None and exclude is not None:
            raise ValueError('Include and exclude cannot both be specified')
        Transform.validate_keys_sequence(include, 'include')
        Transform.validate_keys_sequence(exclude, 'exclude')
        Transform.validate_keys_sequence(label_keys, 'label_keys')
        return (include, exclude)

    @staticmethod
    def validate_keys_sequence(keys: TypeKeys, name: str) -> None:
        """Ensure that the input is not a string but a sequence of strings."""
        if keys is None:
            return
        if isinstance(keys, str):
            message = f'"{name}" must be a sequence of strings, not a string "{keys}"'
            raise ValueError(message)
        if not is_iterable(keys):
            message = f'"{name}" must be a sequence of strings, not {type(keys)}'
            raise ValueError(message)

    @staticmethod
    def nib_to_sitk(data: TypeData, affine: TypeData) -> sitk.Image:
        return nib_to_sitk(data, affine)

    @staticmethod
    def sitk_to_nib(image: sitk.Image) -> TypeDataAffine:
        return sitk_to_nib(image)

    def _get_reproducing_arguments(self):
        """Return a dictionary with the arguments that would be necessary to
        reproduce the transform exactly."""
        reproducing_arguments = {'include': self.include, 'exclude': self.exclude, 'copy': self.copy}
        args_names = {name: getattr(self, name) for name in self.args_names}
        reproducing_arguments.update(args_names)
        return reproducing_arguments

    def is_invertible(self):
        return hasattr(self, 'invert_transform')

    def inverse(self):
        if not self.is_invertible():
            raise RuntimeError(f'{self.name} is not invertible')
        new = copy.deepcopy(self)
        new.invert_transform = not self.invert_transform
        return new

    @staticmethod
    @contextmanager
    def _use_seed(seed):
        """Perform an operation using a specific seed for the PyTorch RNG."""
        torch_rng_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        yield
        torch.random.set_rng_state(torch_rng_state)

    @staticmethod
    def get_sitk_interpolator(interpolation: str) -> int:
        return get_sitk_interpolator(interpolation)

    @staticmethod
    def parse_bounds(bounds_parameters: TypeBounds) -> TypeSixBounds | None:
        if bounds_parameters is None:
            return None
        try:
            bounds_parameters = tuple(bounds_parameters)
        except TypeError:
            bounds_parameters = (bounds_parameters,)
        for number in bounds_parameters:
            if not isinstance(number, (int, np.integer)) or number < 0:
                message = f'Bounds values must be integers greater or equal to zero, not "{bounds_parameters}" of type {type(number)}'
                raise ValueError(message)
        bounds_parameters_tuple = tuple((int(n) for n in bounds_parameters))
        bounds_parameters_length = len(bounds_parameters_tuple)
        if bounds_parameters_length == 6:
            return bounds_parameters_tuple
        if bounds_parameters_length == 1:
            return 6 * bounds_parameters_tuple
        if bounds_parameters_length == 3:
            repeat = np.repeat(bounds_parameters_tuple, 2).tolist()
            return tuple(repeat)
        message = f'Bounds parameter must be an integer or a tuple of 3 or 6 integers, not {bounds_parameters_tuple}'
        raise ValueError(message)

    @staticmethod
    def ones(tensor: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(tensor, dtype=torch.bool)

    @staticmethod
    def mean(tensor: torch.Tensor) -> torch.Tensor:
        mask = tensor > tensor.float().mean()
        return mask

    def get_mask_from_masking_method(self, masking_method: TypeMaskingMethod, subject: Subject, tensor: torch.Tensor, labels: Sequence[int] | None=None) -> torch.Tensor:
        if masking_method is None:
            return self.ones(tensor)
        elif callable(masking_method):
            return masking_method(tensor)
        elif type(masking_method) is str:
            in_subject = masking_method in subject
            if in_subject and isinstance(subject[masking_method], LabelMap):
                if labels is None:
                    return subject[masking_method].data.bool()
                else:
                    mask_data = subject[masking_method].data
                    volumes = [mask_data == label for label in labels]
                    return torch.stack(volumes).sum(0).bool()
            possible_axis = masking_method.capitalize()
            if possible_axis in ANATOMICAL_AXES:
                return self.get_mask_from_anatomical_label(possible_axis, tensor)
        elif type(masking_method) in (tuple, list, int):
            return self.get_mask_from_bounds(masking_method, tensor)
        first_anat_axes = tuple((s[0] for s in ANATOMICAL_AXES))
        message = f'Masking method must be one of:\n 1) A callable object, such as a function\n 2) The name of a label map in the subject ({subject.get_images_names()})\n 3) An anatomical label {ANATOMICAL_AXES + first_anat_axes}\n 4) A bounds parameter (int, tuple of 3 ints, or tuple of 6 ints)\n The passed value, "{masking_method}", of type "{type(masking_method)}", is not valid'
        raise ValueError(message)

    @staticmethod
    def get_mask_from_anatomical_label(anatomical_label: str, tensor: torch.Tensor) -> torch.Tensor:
        anatomical_label = anatomical_label.capitalize()
        if anatomical_label not in ANATOMICAL_AXES:
            message = f'Anatomical label must be one of {ANATOMICAL_AXES} not {anatomical_label}'
            raise ValueError(message)
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        _, width, height, depth = tensor.shape
        if anatomical_label == 'Right':
            mask[:, width // 2:] = True
        elif anatomical_label == 'Left':
            mask[:, :width // 2] = True
        elif anatomical_label == 'Anterior':
            mask[:, :, height // 2:] = True
        elif anatomical_label == 'Posterior':
            mask[:, :, :height // 2] = True
        elif anatomical_label == 'Superior':
            mask[:, :, :, depth // 2:] = True
        elif anatomical_label == 'Inferior':
            mask[:, :, :, :depth // 2] = True
        return mask

    def get_mask_from_bounds(self, bounds_parameters: TypeBounds, tensor: torch.Tensor) -> torch.Tensor:
        bounds_parameters = self.parse_bounds(bounds_parameters)
        assert bounds_parameters is not None
        low = bounds_parameters[::2]
        high = bounds_parameters[1::2]
        i0, j0, k0 = low
        i1, j1, k1 = np.array(tensor.shape[1:]) - high
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        mask[:, i0:i1, j0:j1, k0:k1] = True
        return mask

def _parse_scales_isotropic(scales, isotropic):
    scales = to_tuple(scales)
    if isotropic and len(scales) in (3, 6):
        message = f'If "isotropic" is True, the value for "scales" must have length 1 or 2, but "{scales}" was passed. If you want to set isotropic scaling, use a single value or two values as a range for the scaling factor. Refer to the documentation for more information.'
        raise ValueError(message)

class RandomElasticDeformation(RandomTransform, SpatialTransform):
    """Apply dense random elastic deformation.

    A random displacement is assigned to a coarse grid of control points around
    and inside the image. The displacement at each voxel is interpolated from
    the coarse grid using cubic B-splines.

    The `'Deformable Registration' <https://www.sciencedirect.com/topics/computer-science/deformable-registration>`_
    topic on ScienceDirect contains useful articles explaining interpolation of
    displacement fields using cubic B-splines.

    .. warning:: This transform is slow as it requires expensive computations.
        If your images are large you might want to use
        :class:`~torchio.transforms.RandomAffine` instead.

    Args:
        num_control_points: Number of control points along each dimension of
            the coarse grid :math:`(n_x, n_y, n_z)`.
            If a single value :math:`n` is passed,
            then :math:`n_x = n_y = n_z = n`.
            Smaller numbers generate smoother deformations.
            The minimum number of control points is ``4`` as this transform
            uses cubic B-splines to interpolate displacement.
        max_displacement: Maximum displacement along each dimension at each
            control point :math:`(D_x, D_y, D_z)`.
            The displacement along dimension :math:`i` at each control point is
            :math:`d_i \\sim \\mathcal{U}(0, D_i)`.
            If a single value :math:`D` is passed,
            then :math:`D_x = D_y = D_z = D`.
            Note that the total maximum displacement would actually be
            :math:`D_{max} = \\sqrt{D_x^2 + D_y^2 + D_z^2}`.
        locked_borders: If ``0``, all displacement vectors are kept.
            If ``1``, displacement of control points at the
            border of the coarse grid will be set to ``0``.
            If ``2``, displacement of control points at the border of the image
            (red dots in the image below) will also be set to ``0``.
        image_interpolation: See :ref:`Interpolation`.
            Note that this is the interpolation used to compute voxel
            intensities when resampling using the dense displacement field.
            The value of the dense displacement at each voxel is always
            interpolated with cubic B-splines from the values at the control
            points of the coarse grid.
        label_interpolation: See :ref:`Interpolation`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    `This gist <https://gist.github.com/fepegar/b723d15de620cd2a3a4dbd71e491b59d>`_
    can also be used to better understand the meaning of the parameters.

    This is an example from the
    `3D Slicer registration FAQ <https://www.slicer.org/wiki/Documentation/4.10/FAQ/Registration#What.27s_the_BSpline_Grid_Size.3F>`_.

    .. image:: https://www.slicer.org/w/img_auth.php/6/6f/RegLib_BSplineGridModel.png
        :alt: B-spline example from 3D Slicer documentation

    To generate a similar grid of control points with TorchIO,
    the transform can be instantiated as follows::

        >>> from torchio import RandomElasticDeformation
        >>> transform = RandomElasticDeformation(
        ...     num_control_points=(7, 7, 7),  # or just 7
        ...     locked_borders=2,
        ... )

    Note that control points outside the image bounds are not showed in the
    example image (they would also be red as we set :attr:`locked_borders`
    to ``2``).

    .. warning:: Image folding may occur if the maximum displacement is larger
        than half the coarse grid spacing. The grid spacing can be computed
        using the image bounds in physical space [#]_ and the number of control
        points::

            >>> import numpy as np
            >>> import torchio as tio
            >>> image = tio.datasets.Slicer().MRHead.as_sitk()
            >>> image.GetSize()  # in voxels
            (256, 256, 130)
            >>> image.GetSpacing()  # in mm
            (1.0, 1.0, 1.2999954223632812)
            >>> bounds = np.array(image.GetSize()) * np.array(image.GetSpacing())
            >>> bounds  # mm
            array([256.        , 256.        , 168.99940491])
            >>> num_control_points = np.array((7, 7, 6))
            >>> grid_spacing = bounds / (num_control_points - 2)
            >>> grid_spacing
            array([51.2       , 51.2       , 42.24985123])
            >>> potential_folding = grid_spacing / 2
            >>> potential_folding  # mm
            array([25.6       , 25.6       , 21.12492561])

        Using a :attr:`max_displacement` larger than the computed
        :attr:`potential_folding` will raise a :class:`RuntimeWarning`.

        .. [#] Technically, :math:`2 \\epsilon` should be added to the
            image bounds, where :math:`\\epsilon = 2^{-3}` `according to ITK
            source code <https://github.com/InsightSoftwareConsortium/ITK/blob/633f84548311600845d54ab2463d3412194690a8/Modules/Core/Transform/include/itkBSplineTransformInitializer.hxx#L116-L138>`_.
    """

    def __init__(self, num_control_points: Union[int, TypeTripletInt]=7, max_displacement: Union[float, TypeTripletFloat]=7.5, locked_borders: int=2, image_interpolation: str='linear', label_interpolation: str='nearest', **kwargs):
        super().__init__(**kwargs)
        self._bspline_transformation = None
        self.num_control_points = to_tuple(num_control_points, length=3)
        _parse_num_control_points(self.num_control_points)
        self.max_displacement = to_tuple(max_displacement, length=3)
        _parse_max_displacement(self.max_displacement)
        self.num_locked_borders = locked_borders
        if locked_borders not in (0, 1, 2):
            raise ValueError('locked_borders must be 0, 1, or 2')
        if locked_borders == 2 and 4 in self.num_control_points:
            message = 'Setting locked_borders to 2 and using less than 5 controlpoints results in an identity transform. Lock fewer borders or use more control points.'
            raise ValueError(message)
        self.image_interpolation = self.parse_interpolation(image_interpolation)
        self.label_interpolation = self.parse_interpolation(label_interpolation)

    @staticmethod
    def get_params(num_control_points: TypeTripletInt, max_displacement: tuple[float, float, float], num_locked_borders: int) -> np.ndarray:
        grid_shape = num_control_points
        num_dimensions = 3
        coarse_field = torch.rand(*grid_shape, num_dimensions)
        coarse_field -= 0.5
        coarse_field *= 2
        for dimension in range(3):
            coarse_field[..., dimension] *= max_displacement[dimension]
        for i in range(num_locked_borders):
            coarse_field[i, :] = 0
            coarse_field[-1 - i, :] = 0
            coarse_field[:, i] = 0
            coarse_field[:, -1 - i] = 0
        return coarse_field.numpy()

    def apply_transform(self, subject: Subject) -> Subject:
        subject.check_consistent_spatial_shape()
        control_points = self.get_params(self.num_control_points, self.max_displacement, self.num_locked_borders)
        arguments = {'control_points': control_points, 'max_displacement': self.max_displacement, 'image_interpolation': self.image_interpolation, 'label_interpolation': self.label_interpolation}
        transform = ElasticDeformation(**self.add_base_args(arguments))
        transformed = transform(subject)
        assert isinstance(transformed, Subject)
        return transformed

def _parse_axes(axes: Union[int, tuple[int, ...]]):
    axes_tuple = to_tuple(axes)
    for axis in axes_tuple:
        is_int = isinstance(axis, int)
        is_string = isinstance(axis, str)
        valid_number = is_int and axis in (0, 1, 2)
        if not is_string and (not valid_number):
            message = f'All axes must be 0, 1 or 2, but found "{axis}" with type {type(axis)}'
            raise ValueError(message)
    return axes_tuple

class RandomAnisotropy(RandomTransform):
    """Downsample an image along an axis and upsample to initial space.

    This transform simulates an image that has been acquired using anisotropic
    spacing and resampled back to its original spacing.

    Similar to the work by Billot et al.: `Partial Volume Segmentation of Brain
    MRI Scans of any Resolution and
    Contrast <https://link.springer.com/chapter/10.1007/978-3-030-59728-3_18>`_.

    Args:
        axes: Axis or tuple of axes along which the image will be downsampled.
        downsampling: Downsampling factor :math:`m \\gt 1`. If a tuple
            :math:`(a, b)` is provided then :math:`m \\sim \\mathcal{U}(a, b)`.
        image_interpolation: Image interpolation used to upsample the image
            back to its initial spacing. Downsampling is performed using
            nearest neighbor interpolation. See :ref:`Interpolation` for
            supported interpolation types.
        scalars_only: Apply only to instances of :class:`torchio.ScalarImage`.
            This is useful when the segmentation quality needs to be kept,
            as in `Billot et al. <https://link.springer.com/chapter/10.1007/978-3-030-59728-3_18>`_.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    Example:
        >>> import torchio as tio
        >>> transform = tio.RandomAnisotropy(axes=1, downsampling=2)
        >>> transform = tio.RandomAnisotropy(
        ...     axes=(0, 1, 2),
        ...     downsampling=(2, 5),
        ... )   # Multiply spacing of one of the 3 axes by a factor randomly chosen in [2, 5]
        >>> colin = tio.datasets.Colin27()
        >>> transformed = transform(colin)
    """

    def __init__(self, axes: Union[int, tuple[int, ...]]=(0, 1, 2), downsampling: TypeRangeFloat=(1.5, 5), image_interpolation: str='linear', scalars_only: bool=True, **kwargs):
        super().__init__(**kwargs)
        self.axes = self.parse_axes(axes)
        self.downsampling_range = self._parse_range(downsampling, 'downsampling', min_constraint=1)
        parsed_interpolation = self.parse_interpolation(image_interpolation)
        self.image_interpolation = parsed_interpolation
        self.scalars_only = scalars_only

    def get_params(self, axes: tuple[int, ...], downsampling_range: tuple[float, float]) -> tuple[int, float]:
        axis = axes[torch.randint(0, len(axes), (1,))]
        downsampling = self.sample_uniform(*downsampling_range)
        return (axis, downsampling)

    @staticmethod
    def parse_axes(axes: Union[int, tuple[int, ...]]):
        axes_tuple = to_tuple(axes)
        for axis in axes_tuple:
            is_int = isinstance(axis, int)
            if not is_int or axis not in (0, 1, 2):
                raise ValueError('All axes must be 0, 1 or 2')
        return axes_tuple

    def apply_transform(self, subject: Subject) -> Subject:
        is_2d = subject.get_first_image().is_2d()
        if is_2d and 2 in self.axes:
            warnings.warn(f'Input image is 2D, but "2" is in axes: {self.axes}', RuntimeWarning, stacklevel=2)
            self.axes = list(self.axes)
            self.axes.remove(2)
        axis, downsampling = self.get_params(self.axes, self.downsampling_range)
        target_spacing = list(subject.spacing)
        target_spacing[axis] *= downsampling
        downsample_args = self.add_base_args({'target': tuple(target_spacing), 'image_interpolation': 'nearest', 'scalars_only': self.scalars_only})
        image = subject.get_first_image()
        upsample_args = self.add_base_args({'target': (image.spatial_shape, image.affine), 'image_interpolation': self.image_interpolation, 'scalars_only': self.scalars_only})
        downsample = Resample(**downsample_args)
        downsampled = downsample(subject)
        upsample = Resample(**upsample_args)
        upsampled = upsample(downsampled)
        assert isinstance(upsampled, Subject)
        return upsampled

class RandomSwap(RandomTransform, IntensityTransform):
    """Randomly swap patches within an image.

    This is typically used in `context restoration for self-supervised learning
    <https://www.sciencedirect.com/science/article/pii/S1361841518304699>`_.

    Args:
        patch_size: Tuple of integers :math:`(w, h, d)` to swap patches
            of size :math:`w \\times h \\times d`.
            If a single number :math:`n` is provided, :math:`w = h = d = n`.
        num_iterations: Number of times that two patches will be swapped.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """

    def __init__(self, patch_size: TypeTuple=15, num_iterations: int=100, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = np.array(to_tuple(patch_size))
        self.num_iterations = self._parse_num_iterations(num_iterations)

    @staticmethod
    def _parse_num_iterations(num_iterations):
        if not isinstance(num_iterations, int):
            raise TypeError(f'num_iterations must be an int,not {num_iterations}')
        if num_iterations < 0:
            raise ValueError(f'num_iterations must be positive,not {num_iterations}')
        return num_iterations

    @staticmethod
    def get_params(tensor: torch.Tensor, patch_size: np.ndarray, num_iterations: int) -> list[tuple[TypeTripletInt, TypeTripletInt]]:
        si, sj, sk = tensor.shape[-3:]
        spatial_shape = (si, sj, sk)
        locations = []
        for _ in range(num_iterations):
            first_ini, first_fin = get_random_indices_from_shape(spatial_shape, patch_size.tolist())
            while True:
                second_ini, second_fin = get_random_indices_from_shape(spatial_shape, patch_size.tolist())
                larger_than_initial = np.all(second_ini >= first_ini)
                less_than_final = np.all(second_fin <= first_fin)
                if larger_than_initial and less_than_final:
                    continue
                else:
                    break
            location = (tuple(first_ini), tuple(second_ini))
            locations.append(location)
        return locations

    def apply_transform(self, subject: Subject) -> Subject:
        images_dict = self.get_images_dict(subject)
        if not images_dict:
            return subject
        arguments: dict[str, dict] = defaultdict(dict)
        for name, image in images_dict.items():
            locations = self.get_params(image.data, self.patch_size, self.num_iterations)
            arguments['locations'][name] = locations
            arguments['patch_size'][name] = self.patch_size
        transform = Swap(**self.add_base_args(arguments))
        transformed = transform(subject)
        assert isinstance(transformed, Subject)
        return transformed

class Gamma(IntensityTransform):
    """Change contrast of an image by raising its values to the power
    :math:`\\gamma`.

    Args:
        gamma: Exponent to which values in the image will be raised.
            Negative and positive values for this argument perform gamma
            compression and expansion, respectively.
            See the `Gamma correction`_ Wikipedia entry for more information.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    .. _Gamma correction: https://en.wikipedia.org/wiki/Gamma_correction

    .. note:: Fractional exponentiation of negative values is generally not
        well-defined for non-complex numbers.
        If negative values are found in the input image :math:`I`,
        the applied transform is :math:`\\text{sign}(I) |I|^\\gamma`,
        instead of the usual :math:`I^\\gamma`. The
        :class:`~torchio.transforms.preprocessing.intensity.rescale.RescaleIntensity`
        transform may be used to ensure that all values are positive. This is
        generally not problematic, but it is recommended to visualize results
        on image with negative values. More information can be found on
        `this StackExchange question`_.

        .. _this StackExchange question: https://math.stackexchange.com/questions/317528/how-do-you-compute-negative-numbers-to-fractional-powers

    Example:
        >>> import torchio as tio
        >>> subject = tio.datasets.FPG()
        >>> transform = tio.Gamma(0.8)
        >>> transformed = transform(subject)
    """

    def __init__(self, gamma: float, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.args_names = ['gamma']
        self.invert_transform = False

    def apply_transform(self, subject: Subject) -> Subject:
        gamma = self.gamma
        for name, image in self.get_images_dict(subject).items():
            if self.arguments_are_dict():
                assert isinstance(self.gamma, dict)
                gamma = self.gamma[name]
            gammas = to_tuple(gamma, length=len(image.data))
            transformed_tensors = []
            image.set_data(image.data.float())
            for gamma, tensor in zip(gammas, image.data):
                if self.invert_transform:
                    correction = power(tensor, 1 - gamma)
                    transformed_tensor = tensor * correction
                else:
                    transformed_tensor = power(tensor, gamma)
                transformed_tensors.append(transformed_tensor)
            image.set_data(torch.stack(transformed_tensors))
        return subject

class Resize(SpatialTransform):
    """Resample images so the output shape matches the given target shape.

    The field of view remains the same.

    .. warning:: In most medical image applications, this transform should not
        be used as it will deform the physical object by scaling anisotropically
        along the different dimensions. The solution to change an image size is
        typically applying :class:`~torchio.transforms.Resample` and
        :class:`~torchio.transforms.CropOrPad`.

    Args:
        target_shape: Tuple :math:`(W, H, D)`. If a single value :math:`N` is
            provided, then :math:`W = H = D = N`. The size of dimensions set to
            -1 will be kept.
        image_interpolation: See :ref:`Interpolation`.
        label_interpolation: See :ref:`Interpolation`.
    """

    def __init__(self, target_shape: TypeSpatialShape, image_interpolation: str='linear', label_interpolation: str='nearest', **kwargs):
        super().__init__(**kwargs)
        self.target_shape = np.asarray(to_tuple(target_shape, length=3))
        self.image_interpolation = self.parse_interpolation(image_interpolation)
        self.label_interpolation = self.parse_interpolation(label_interpolation)
        self.args_names = ['target_shape', 'image_interpolation', 'label_interpolation']

    def apply_transform(self, subject: Subject) -> Subject:
        shape_in = np.asarray(subject.spatial_shape)
        shape_out = self.target_shape
        negative_mask = shape_out == -1
        shape_out[negative_mask] = shape_in[negative_mask]
        spacing_in = np.asarray(subject.spacing)
        spacing_out = shape_in / shape_out * spacing_in
        resample = Resample(spacing_out, image_interpolation=self.image_interpolation, label_interpolation=self.label_interpolation, **self.get_base_args())
        resampled = resample(subject)
        assert isinstance(resampled, Subject)
        if not resampled.spatial_shape == tuple(shape_out):
            message = f'Output shape {resampled.spatial_shape} != target shape {tuple(shape_out)}. Fixing with CropOrPad'
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            crop_pad = CropOrPad(shape_out, **self.get_base_args())
            resampled = crop_pad(resampled)
        assert isinstance(resampled, Subject)
        return resampled

class CropOrPad(SpatialTransform):
    """Modify the field of view by cropping or padding to match a target shape.

    This transform modifies the affine matrix associated to the volume so that
    physical positions of the voxels are maintained.

    Args:
        target_shape: Tuple :math:`(W, H, D)`. If a single value :math:`N` is
            provided, then :math:`W = H = D = N`. If ``None``, the shape will
            be computed from the :attr:`mask_name` (and the :attr:`labels`, if
            :attr:`labels` is not ``None``).
        padding_mode: Same as :attr:`padding_mode` in
            :class:`~torchio.transforms.Pad`.
        mask_name: If ``None``, the centers of the input and output volumes
            will be the same.
            If a string is given, the output volume center will be the center
            of the bounding box of non-zero values in the image named
            :attr:`mask_name`.
        labels: If a label map is used to generate the mask, sequence of labels
            to consider.
        only_crop: If ``True``, padding will not be applied, only cropping will
            be done. ``only_crop`` and ``only_pad`` cannot both be ``True``.
        only_pad: If ``True``, cropping will not be applied, only padding will
            be done. ``only_crop`` and ``only_pad`` cannot both be ``True``.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    Example:
        >>> import torchio as tio
        >>> subject = tio.Subject(
        ...     chest_ct=tio.ScalarImage('subject_a_ct.nii.gz'),
        ...     heart_mask=tio.LabelMap('subject_a_heart_seg.nii.gz'),
        ... )
        >>> subject.chest_ct.shape
        torch.Size([1, 512, 512, 289])
        >>> transform = tio.CropOrPad(
        ...     (120, 80, 180),
        ...     mask_name='heart_mask',
        ... )
        >>> transformed = transform(subject)
        >>> transformed.chest_ct.shape
        torch.Size([1, 120, 80, 180])

    .. warning:: If :attr:`target_shape` is ``None``, subjects in the dataset
        will probably have different shapes. This is probably fine if you are
        using `patch-based training <https://docs.torchio.org/patches/index.html>`_.
        If you are using full volumes for training and a batch size larger than
        one, an error will be raised by the :class:`~torch.utils.data.DataLoader`
        while trying to collate the batches.

    .. plot::

        import torchio as tio
        t1 = tio.datasets.Colin27().t1
        crop_pad = tio.CropOrPad((512, 512, 32))
        t1_pad_crop = crop_pad(t1)
        subject = tio.Subject(t1=t1, crop_pad=t1_pad_crop)
        subject.plot()
    """

    def __init__(self, target_shape: int | TypeTripletInt | None=None, padding_mode: str | float=0, mask_name: str | None=None, labels: Sequence[int] | None=None, only_crop: bool=False, only_pad: bool=False, **kwargs):
        if target_shape is None and mask_name is None:
            message = 'If mask_name is None, a target shape must be passed'
            raise ValueError(message)
        super().__init__(**kwargs)
        if target_shape is None:
            self.target_shape = None
        else:
            self.target_shape = parse_spatial_shape(target_shape)
        self.padding_mode = padding_mode
        if mask_name is not None and (not isinstance(mask_name, str)):
            message = f'If mask_name is not None, it must be a string, not {type(mask_name)}'
            raise ValueError(message)
        if mask_name is None:
            if labels is not None:
                message = f'If mask_name is None, labels should be None, but "{labels}" was passed'
                raise ValueError(message)
            self.compute_crop_or_pad = self._compute_center_crop_or_pad
        else:
            if not isinstance(mask_name, str):
                message = f'If mask_name is not None, it must be a string, not {type(mask_name)}'
                raise ValueError(message)
            self.compute_crop_or_pad = self._compute_mask_center_crop_or_pad
        self.mask_name = mask_name
        self.labels = labels
        if only_pad and only_crop:
            message = 'only_crop and only_pad cannot both be True'
            raise ValueError(message)
        self.only_crop = only_crop
        self.only_pad = only_pad
        self.args_names = ['target_shape', 'padding_mode', 'mask_name', 'labels', 'only_crop', 'only_pad']

    @staticmethod
    def _bbox_mask(mask_volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return 6 coordinates of a 3D bounding box from a given mask.

        Taken from `this SO question <https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array>`_.

        Args:
            mask_volume: 3D NumPy array.
        """
        i_any = np.any(mask_volume, axis=(1, 2))
        j_any = np.any(mask_volume, axis=(0, 2))
        k_any = np.any(mask_volume, axis=(0, 1))
        i_min, i_max = np.where(i_any)[0][[0, -1]]
        j_min, j_max = np.where(j_any)[0][[0, -1]]
        k_min, k_max = np.where(k_any)[0][[0, -1]]
        bb_min = np.array([i_min, j_min, k_min])
        bb_max = np.array([i_max, j_max, k_max]) + 1
        return (bb_min, bb_max)

    @staticmethod
    def _get_six_bounds_parameters(parameters: np.ndarray) -> TypeSixBounds:
        """Compute bounds parameters for ITK filters.

        Args:
            parameters: Tuple :math:`(w, h, d)` with the number of voxels to be
                cropped or padded.

        Returns:
            Tuple :math:`(w_{ini}, w_{fin}, h_{ini}, h_{fin}, d_{ini}, d_{fin})`,
            where :math:`n_{ini} = \\left \\lceil \\frac{n}{2} \\right \\rceil` and
            :math:`n_{fin} = \\left \\lfloor \\frac{n}{2} \\right \\rfloor`.

        Example:
            >>> p = np.array((4, 0, 7))
            >>> CropOrPad._get_six_bounds_parameters(p)
            (2, 2, 0, 0, 4, 3)
        """
        parameters = parameters / 2
        result = []
        for number in parameters:
            ini, fin = (int(np.ceil(number)), int(np.floor(number)))
            result.extend([ini, fin])
        i1, i2, j1, j2, k1, k2 = result
        return (i1, i2, j1, j2, k1, k2)

    def _compute_cropping_padding_from_shapes(self, source_shape: TypeTripletInt) -> tuple[TypeSixBounds | None, TypeSixBounds | None]:
        diff_shape = np.array(self.target_shape) - source_shape
        cropping = -np.minimum(diff_shape, 0)
        if cropping.any():
            cropping_params = self._get_six_bounds_parameters(cropping)
        else:
            cropping_params = None
        padding = np.maximum(diff_shape, 0)
        if padding.any():
            padding_params = self._get_six_bounds_parameters(padding)
        else:
            padding_params = None
        return (padding_params, cropping_params)

    def _compute_center_crop_or_pad(self, subject: Subject) -> tuple[TypeSixBounds | None, TypeSixBounds | None]:
        source_shape = subject.spatial_shape
        parameters = self._compute_cropping_padding_from_shapes(source_shape)
        padding_params, cropping_params = parameters
        return (padding_params, cropping_params)

    def _compute_mask_center_crop_or_pad(self, subject: Subject) -> tuple[TypeSixBounds | None, TypeSixBounds | None]:
        if self.mask_name not in subject:
            message = f'Mask name "{self.mask_name}" not found in subject keys "{tuple(subject.keys())}". Using volume center instead'
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            return self._compute_center_crop_or_pad(subject=subject)
        mask_data = self.get_mask_from_masking_method(self.mask_name, subject, subject[self.mask_name].data, self.labels).numpy()
        if not np.any(mask_data):
            message = f'All values found in the mask "{self.mask_name}" are zero. Using volume center instead'
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            return self._compute_center_crop_or_pad(subject=subject)
        subject_shape = subject.spatial_shape
        bb_min, bb_max = self._bbox_mask(mask_data[0])
        center_mask = np.mean((bb_min, bb_max), axis=0)
        padding = []
        cropping = []
        if self.target_shape is None:
            target_shape = bb_max - bb_min
        else:
            target_shape = self.target_shape
        for dim in range(3):
            target_dim = target_shape[dim]
            center_dim = center_mask[dim]
            subject_dim = subject_shape[dim]
            center_on_index = not center_dim % 1
            target_even = not target_dim % 2
            if target_even ^ center_on_index:
                center_dim -= 0.5
            begin = center_dim - target_dim / 2
            if begin >= 0:
                crop_ini = begin
                pad_ini = 0
            else:
                crop_ini = 0
                pad_ini = -begin
            end = center_dim + target_dim / 2
            if end <= subject_dim:
                crop_fin = subject_dim - end
                pad_fin = 0
            else:
                crop_fin = 0
                pad_fin = end - subject_dim
            padding.extend([pad_ini, pad_fin])
            cropping.extend([crop_ini, crop_fin])
        padding_array = np.asarray(padding, dtype=int)
        cropping_array = np.asarray(cropping, dtype=int)
        if padding_array.any():
            padding_params = tuple(padding_array.tolist())
        else:
            padding_params = None
        if cropping_array.any():
            cropping_params = tuple(cropping_array.tolist())
        else:
            cropping_params = None
        return (padding_params, cropping_params)

    def apply_transform(self, subject: Subject) -> Subject:
        subject.check_consistent_space()
        padding_params, cropping_params = self.compute_crop_or_pad(subject)
        padding_kwargs = {'padding_mode': self.padding_mode}
        if padding_params is not None and (not self.only_crop):
            pad = Pad(padding_params, **self.get_base_args(), **padding_kwargs)
            subject = pad(subject)
        if cropping_params is not None and (not self.only_pad):
            crop = Crop(cropping_params, **self.get_base_args())
            subject = crop(subject)
        return subject

class EnsureShapeMultiple(SpatialTransform):
    """Ensure that all values in the image shape are divisible by :math:`n`.

    Some convolutional neural network architectures need that the size of the
    input across all spatial dimensions is a power of :math:`2`.

    For example, the canonical 3D U-Net from
    `Çiçek et al. <https://link.springer.com/chapter/10.1007/978-3-319-46723-8_49>`_
    includes three downsampling (pooling) and upsampling operations:

    .. image:: https://www.researchgate.net/profile/Olaf-Ronneberger/publication/304226155/figure/fig1/AS:375619658502144@1466566113191/The-3D-u-net-architecture-Blue-boxes-represent-feature-maps-The-number-of-channels-is.png
        :alt: 3D U-Net

    Pooling operations in PyTorch round down the output size:

        >>> import torch
        >>> x = torch.rand(3, 10, 20, 31)
        >>> x_down = torch.nn.functional.max_pool3d(x, 2)
        >>> x_down.shape
        torch.Size([3, 5, 10, 15])

    If we upsample this tensor, the original shape is lost:

        >>> x_down_up = torch.nn.functional.interpolate(x_down, scale_factor=2)
        >>> x_down_up.shape
        torch.Size([3, 10, 20, 30])
        >>> x.shape
        torch.Size([3, 10, 20, 31])

    If we try to concatenate ``x_down`` and ``x_down_up`` (to create skip
    connections), we will get an error. It is therefore good practice to ensure
    that the size of our images is such that concatenations will be safe.

    .. note:: In these examples, it's assumed that all convolutions in the
        U-Net use padding so that the output size is the same as the input
        size.

    The image above shows :math:`3` downsampling operations, so the input size
    along all dimensions should be a multiple of :math:`2^3 = 8`.

    Example (assuming ``pip install unet`` has been run before):

        >>> import torchio as tio
        >>> import unet
        >>> net = unet.UNet3D(padding=1)
        >>> t1 = tio.datasets.Colin27().t1
        >>> tensor_bad = t1.data.unsqueeze(0)
        >>> tensor_bad.shape
        torch.Size([1, 1, 181, 217, 181])
        >>> net(tensor_bad).shape
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          File "/home/fernando/miniconda3/envs/resseg/lib/python3.7/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
            result = self.forward(*input, **kwargs)
          File "/home/fernando/miniconda3/envs/resseg/lib/python3.7/site-packages/unet/unet.py", line 122, in forward
            x = self.decoder(skip_connections, encoding)
          File "/home/fernando/miniconda3/envs/resseg/lib/python3.7/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
            result = self.forward(*input, **kwargs)
          File "/home/fernando/miniconda3/envs/resseg/lib/python3.7/site-packages/unet/decoding.py", line 61, in forward
            x = decoding_block(skip_connection, x)
          File "/home/fernando/miniconda3/envs/resseg/lib/python3.7/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
            result = self.forward(*input, **kwargs)
          File "/home/fernando/miniconda3/envs/resseg/lib/python3.7/site-packages/unet/decoding.py", line 131, in forward
            x = torch.cat((skip_connection, x), dim=CHANNELS_DIMENSION)
        RuntimeError: Sizes of tensors must match except in dimension 1. Got 45 and 44 in dimension 2 (The offending index is 1)
        >>> num_poolings = 3
        >>> fix_shape_unet = tio.EnsureShapeMultiple(2**num_poolings)
        >>> t1_fixed = fix_shape_unet(t1)
        >>> tensor_ok = t1_fixed.data.unsqueeze(0)
        >>> tensor_ok.shape
        torch.Size([1, 1, 184, 224, 184])  # as expected

    Args:
        target_multiple: Tuple :math:`(n_w, n_h, n_d)`, so that the size of the
            output along axis :math:`i` is a multiple of :math:`n_i`. If a
            single value :math:`n` is provided, then
            :math:`n_w = n_h = n_d = n`.
        method: Either ``'crop'`` or ``'pad'``.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    Example:
        >>> import torchio as tio
        >>> image = tio.datasets.Colin27().t1
        >>> image.shape
        (1, 181, 217, 181)
        >>> transform = tio.EnsureShapeMultiple(8, method='pad')
        >>> transformed = transform(image)
        >>> transformed.shape
        (1, 184, 224, 184)
        >>> transform = tio.EnsureShapeMultiple(8, method='crop')
        >>> transformed = transform(image)
        >>> transformed.shape
        (1, 176, 216, 176)
        >>> image_2d = image.data[..., :1]
        >>> image_2d.shape
        torch.Size([1, 181, 217, 1])
        >>> transformed = transform(image_2d)
        >>> transformed.shape
        torch.Size([1, 176, 216, 1])
    """

    def __init__(self, target_multiple: int | TypeTripletInt, *, method: str='pad', **kwargs):
        super().__init__(**kwargs)
        self.target_multiple = np.array(to_tuple(target_multiple, 3))
        if method not in ('crop', 'pad'):
            raise ValueError('Method must be "crop" or "pad"')
        self.method = method

    def apply_transform(self, subject: Subject) -> Subject:
        source_shape = np.array(subject.spatial_shape, np.uint16)
        function: Callable = np.floor if self.method == 'crop' else np.ceil
        integer_ratio = function(source_shape / self.target_multiple)
        target_shape = integer_ratio * self.target_multiple
        target_shape = np.maximum(target_shape, 1)
        transform = CropOrPad(target_shape.astype(int), **self.get_base_args())
        subject = transform(subject)
        return subject

