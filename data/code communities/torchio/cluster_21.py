# Cluster 21

def get_sitk_metadata_from_ras_affine(affine: np.ndarray, is_2d: bool=False, lps: bool=True) -> tuple[TypeTripletFloat, TypeTripletFloat, TypeDirection]:
    direction_ras, spacing_array = get_rotation_and_spacing_from_affine(affine)
    origin_ras = affine[:3, 3]
    origin_lps = np.dot(FLIPXY_33, origin_ras)
    direction_lps = np.dot(FLIPXY_33, direction_ras)
    if is_2d:
        direction_lps = np.diag((-1, -1)).astype(np.float64)
        direction_ras = np.diag((1, 1)).astype(np.float64)
    origin_array = origin_lps if lps else origin_ras
    direction_array = direction_lps if lps else direction_ras
    direction_array = direction_array.flatten()
    ox, oy, oz = origin_array
    sx, sy, sz = spacing_array
    direction: TypeDirection
    if is_2d:
        d1, d2, d3, d4 = direction_array
        direction = (d1, d2, d3, d4)
    else:
        d1, d2, d3, d4, d5, d6, d7, d8, d9 = direction_array
        direction = (d1, d2, d3, d4, d5, d6, d7, d8, d9)
    origin = (ox, oy, oz)
    spacing = (sx, sy, sz)
    return (origin, spacing, direction)

def get_rotation_and_spacing_from_affine(affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation_zoom = affine[:3, :3]
    spacing = np.sqrt(np.sum(rotation_zoom * rotation_zoom, axis=0))
    rotation = rotation_zoom / spacing
    return (rotation, spacing)

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

class Resample(SpatialTransform):
    """Resample image to a different physical space.

    This is a powerful transform that can be used to change the image shape
    or spatial metadata, or to apply a spatial transformation.

    Args:
        target: Argument to define the output space. Can be one of:

            - Output spacing :math:`(s_w, s_h, s_d)`, in mm. If only one value
              :math:`s` is specified, then :math:`s_w = s_h = s_d = s`.

            - Path to an image that will be used as reference.

            - Instance of :class:`~torchio.Image`.

            - Name of an image key in the subject.

            - Tuple ``(spatial_shape, affine)`` defining the output space.

        pre_affine_name: Name of the *image key* (not subject key) storing an
            affine matrix that will be applied to the image header before
            resampling. If ``None``, the image is resampled with an identity
            transform. See usage in the example below.
        image_interpolation: See :ref:`Interpolation`.
        label_interpolation: See :ref:`Interpolation`.
        scalars_only: Apply only to instances of :class:`~torchio.ScalarImage`.
            Used internally by :class:`~torchio.transforms.RandomAnisotropy`.
        antialias: If ``True``, apply Gaussian smoothing before
            downsampling along any dimension that will be downsampled. For example,
            if the input image has spacing (0.5, 0.5, 4) and the target
            spacing is (1, 1, 1), the image will be smoothed along the first two
            dimensions before resampling. Label maps are not smoothed.
            The standard deviations of the Gaussian kernels are computed according to
            the method described in Cardoso et al.,
            `Scale factor point spread function matching: beyond aliasing in image
            resampling
            <https://link.springer.com/chapter/10.1007/978-3-319-24571-3_81>`_,
            MICCAI 2015.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    Example:
        >>> import torch
        >>> import torchio as tio
        >>> transform = tio.Resample()                      # resample all images to 1mm isotropic
        >>> transform = tio.Resample(2)                     # resample all images to 2mm isotropic
        >>> transform = tio.Resample('t1')                  # resample all images to 't1' image space
        >>> # Example: using a precomputed transform to MNI space
        >>> ref_path = tio.datasets.Colin27().t1.path  # this image is in the MNI space, so we can use it as reference/target
        >>> affine_matrix = tio.io.read_matrix('transform_to_mni.txt')  # from a NiftyReg registration. Would also work with e.g. .tfm from SimpleITK
        >>> image = tio.ScalarImage(tensor=torch.rand(1, 256, 256, 180), to_mni=affine_matrix)  # 'to_mni' is an arbitrary name
        >>> transform = tio.Resample(colin.t1.path, pre_affine_name='to_mni')  # nearest neighbor interpolation is used for label maps
        >>> transformed = transform(image)  # "image" is now in the MNI space

    .. note::
        The ``antialias`` option is recommended when large (e.g. > 2×) downsampling
        factors are expected, particularly for offline (before training) preprocessing,
        when run times are not a concern.

    .. plot::

        import torchio as tio
        subject = tio.datasets.FPG()
        subject.remove_image('seg')
        resample = tio.Resample(8)
        t1_resampled = resample(subject.t1)
        subject.add_image(t1_resampled, 'Antialias off')
        resample = tio.Resample(8, antialias=True)
        t1_resampled_antialias = resample(subject.t1)
        subject.add_image(t1_resampled_antialias, 'Antialias on')
        subject.plot()
    """

    def __init__(self, target: TypeTarget=ONE_MILLIMITER_ISOTROPIC, image_interpolation: str='linear', label_interpolation: str='nearest', pre_affine_name: str | None=None, scalars_only: bool=False, antialias: bool=False, **kwargs):
        super().__init__(**kwargs)
        self.target = target
        self.image_interpolation = self.parse_interpolation(image_interpolation)
        self.label_interpolation = self.parse_interpolation(label_interpolation)
        self.pre_affine_name = pre_affine_name
        self.scalars_only = scalars_only
        self.antialias = antialias
        self.args_names = ['target', 'image_interpolation', 'label_interpolation', 'pre_affine_name', 'scalars_only', 'antialias']

    @staticmethod
    def _parse_spacing(spacing: TypeSpacing) -> tuple[float, float, float]:
        result: Iterable
        if isinstance(spacing, Iterable) and len(spacing) == 3:
            result = spacing
        elif isinstance(spacing, Number):
            result = 3 * (spacing,)
        else:
            message = f'Target must be a string, a positive number or a sequence of positive numbers, not {type(spacing)}'
            raise ValueError(message)
        if np.any(np.array(spacing) <= 0):
            message = f'Spacing must be strictly positive, not "{spacing}"'
            raise ValueError(message)
        return result

    @staticmethod
    def check_affine(affine_name: str, image: Image):
        if not isinstance(affine_name, str):
            message = f'Affine name argument must be a string, not {type(affine_name)}'
            raise TypeError(message)
        if affine_name in image:
            matrix = image[affine_name]
            if not isinstance(matrix, (np.ndarray, torch.Tensor)):
                message = f'The affine matrix must be a NumPy array or PyTorch tensor, not {type(matrix)}'
                raise TypeError(message)
            if matrix.shape != (4, 4):
                message = f'The affine matrix shape must be (4, 4), not {matrix.shape}'
                raise ValueError(message)

    @staticmethod
    def check_affine_key_presence(affine_name: str, subject: Subject):
        for image in subject.get_images(intensity_only=False):
            if affine_name in image:
                return
        message = f'An affine name was given ("{affine_name}"), but it was not found in any image in the subject'
        raise ValueError(message)

    def apply_transform(self, subject: Subject) -> Subject:
        use_pre_affine = self.pre_affine_name is not None
        if use_pre_affine:
            assert self.pre_affine_name is not None
            self.check_affine_key_presence(self.pre_affine_name, subject)
        for image in self.get_images(subject):
            if self.target is image:
                continue
            try:
                target_image = subject[self.target]
                if target_image is image:
                    continue
            except (KeyError, TypeError, RuntimeError):
                pass
            if not isinstance(image, ScalarImage):
                if self.scalars_only:
                    continue
                interpolation = self.label_interpolation
            else:
                interpolation = self.image_interpolation
            interpolator = self.get_sitk_interpolator(interpolation)
            if use_pre_affine and self.pre_affine_name in image:
                assert self.pre_affine_name is not None
                self.check_affine(self.pre_affine_name, image)
                matrix = image[self.pre_affine_name]
                if isinstance(matrix, torch.Tensor):
                    matrix = matrix.numpy()
                image.affine = matrix @ image.affine
            floating_sitk = image.as_sitk(force_3d=True)
            resampler = self._get_resampler(interpolator, floating_sitk, subject, self.target)
            if self.antialias and isinstance(image, ScalarImage):
                downsampling_factor = self._get_downsampling_factor(floating_sitk, resampler)
                sigmas = self._get_sigmas(downsampling_factor, floating_sitk.GetSpacing())
                floating_sitk = self._smooth(floating_sitk, sigmas)
            resampled = resampler.Execute(floating_sitk)
            array, affine = sitk_to_nib(resampled)
            image.set_data(torch.as_tensor(array))
            image.affine = affine
        return subject

    @staticmethod
    def _smooth(image: sitk.Image, sigmas: np.ndarray, epsilon: float=1e-09) -> sitk.Image:
        """Smooth the image with a Gaussian kernel.

        Args:
            image: Image to be smoothed.
            sigmas: Standard deviations of the Gaussian kernel for each
                dimension. If a value is NaN, no smoothing is applied in that
                dimension.
            epsilon: Small value to replace NaN values in sigmas, to avoid
                division-by-zero errors.
        """
        sigmas[np.isnan(sigmas)] = epsilon
        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian.SetSigma(sigmas.tolist())
        smoothed = gaussian.Execute(image)
        return smoothed

    @staticmethod
    def _get_downsampling_factor(floating: sitk.Image, resampler: sitk.ResampleImageFilter) -> np.ndarray:
        """Get the downsampling factor for each dimension.

        The downsampling factor is the ratio between the output spacing and
        the input spacing. If the output spacing is smaller than the input
        spacing, the factor is set to NaN, meaning downsampling is not applied
        in that dimension.

        Args:
            floating: The input image to be resampled.
            resampler: The resampler that will be used to resample the image.
        """
        input_spacing = np.array(floating.GetSpacing())
        output_spacing = np.array(resampler.GetOutputSpacing())
        factors = output_spacing / input_spacing
        no_downsampling = factors <= 1
        factors[no_downsampling] = np.nan
        return factors

    def _get_resampler(self, interpolator: int, floating: sitk.Image, subject: Subject, target: TypeTarget) -> sitk.ResampleImageFilter:
        """Instantiate a SimpleITK resampler."""
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(interpolator)
        self._set_resampler_reference(resampler, target, floating, subject)
        return resampler

    def _set_resampler_reference(self, resampler: sitk.ResampleImageFilter, target: TypeSpacing | TypePath | Image, floating_sitk, subject):
        if isinstance(target, (str, Path, Image)):
            if isinstance(target, Image):
                image = target
            elif Path(target).is_file():
                path = target
                image = ScalarImage(path)
            else:
                try:
                    image = subject[target]
                except KeyError as error:
                    message = f'Image name "{target}" not found in subject. If "{target}" is a path, it does not exist or permission has been denied'
                    raise ValueError(message) from error
            self._set_resampler_from_shape_affine(resampler, image.spatial_shape, image.affine)
        elif isinstance(target, Number):
            self._set_resampler_from_spacing(resampler, target, floating_sitk)
        elif isinstance(target, Iterable) and len(target) == 2:
            assert not isinstance(target, str)
            shape, affine = target
            if not (isinstance(shape, Sized) and len(shape) == 3):
                message = f'Target shape must be a sequence of three integers, but "{shape}" was passed'
                raise RuntimeError(message)
            if not affine.shape == (4, 4):
                message = f'Target affine must have shape (4, 4) but the following was passed:\n{shape}'
                raise RuntimeError(message)
            self._set_resampler_from_shape_affine(resampler, shape, affine)
        elif isinstance(target, Iterable) and len(target) == 3:
            self._set_resampler_from_spacing(resampler, target, floating_sitk)
        else:
            raise RuntimeError(f'Target not understood: "{target}"')

    def _set_resampler_from_shape_affine(self, resampler, shape, affine):
        origin, spacing, direction = get_sitk_metadata_from_ras_affine(affine)
        resampler.SetOutputDirection(direction)
        resampler.SetOutputOrigin(origin)
        resampler.SetOutputSpacing(spacing)
        resampler.SetSize(shape)

    def _set_resampler_from_spacing(self, resampler, target, floating_sitk):
        target_spacing = self._parse_spacing(target)
        reference_image = self.get_reference_image(floating_sitk, target_spacing)
        resampler.SetReferenceImage(reference_image)

    @staticmethod
    def get_reference_image(floating_sitk: sitk.Image, spacing: TypeTripletFloat) -> sitk.Image:
        old_spacing = np.array(floating_sitk.GetSpacing(), dtype=float)
        new_spacing = np.array(spacing, dtype=float)
        old_size = np.array(floating_sitk.GetSize())
        old_last_index = old_size - 1
        old_last_index_lps = np.array(floating_sitk.TransformIndexToPhysicalPoint(old_last_index.tolist()), dtype=float)
        old_origin_lps = np.array(floating_sitk.GetOrigin(), dtype=float)
        center_lps = (old_last_index_lps + old_origin_lps) / 2
        new_size = np.floor(old_size * old_spacing / new_spacing)
        new_size[old_size == 1] = 1
        direction = np.asarray(floating_sitk.GetDirection(), dtype=float).reshape(3, 3)
        half_extent = (new_size - 1) / 2 * new_spacing
        new_origin_lps = (center_lps - direction @ half_extent).tolist()
        reference = sitk.Image(new_size.astype(int).tolist(), floating_sitk.GetPixelID(), floating_sitk.GetNumberOfComponentsPerPixel())
        reference.SetDirection(floating_sitk.GetDirection())
        reference.SetSpacing(new_spacing.tolist())
        reference.SetOrigin(new_origin_lps)
        return reference

    @staticmethod
    def _get_sigmas(downsampling_factor: np.ndarray, spacing: np.ndarray) -> np.ndarray:
        """Compute optimal standard deviation for Gaussian kernel.

        From Cardoso et al., `Scale factor point spread function matching:
        beyond aliasing in image resampling
        <https://link.springer.com/chapter/10.1007/978-3-319-24571-3_81>`_,
        MICCAI 2015.

        Args:
            downsampling_factor: Array with the downsampling factor for each
                dimension.
            spacing: Array with the spacing of the input image in mm.
        """
        k = downsampling_factor
        variance = (k ** 2 - 1) * (2 * np.sqrt(2 * np.log(2))) ** (-2)
        sigma = spacing * np.sqrt(variance)
        return sigma

