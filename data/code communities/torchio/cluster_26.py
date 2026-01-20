# Cluster 26

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

def is_iterable(object: Any) -> bool:
    try:
        iter(object)
        return True
    except TypeError:
        return False

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

