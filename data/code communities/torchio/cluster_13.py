# Cluster 13

class HistogramStandardization(NormalizationTransform):
    """Perform histogram standardization of intensity values.

    Implementation of `New variants of a method of MRI scale
    standardization <https://ieeexplore.ieee.org/document/836373>`_.

    See example in :func:`torchio.transforms.HistogramStandardization.train`.

    Args:
        landmarks: Dictionary (or path to a PyTorch file with ``.pt`` or ``.pth``
            extension in which a dictionary has been saved) whose keys are
            image names in the subject and values are NumPy arrays or paths to
            NumPy arrays defining the landmarks after training with
            :meth:`torchio.transforms.HistogramStandardization.train`.
        masking_method: See
            :class:`~torchio.transforms.preprocessing.intensity.NormalizationTransform`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    Example:
        >>> import torch
        >>> import torchio as tio
        >>> landmarks = {
        ...     't1': 't1_landmarks.npy',
        ...     't2': 't2_landmarks.npy',
        ... }
        >>> transform = tio.HistogramStandardization(landmarks)
        >>> torch.save(landmarks, 'path_to_landmarks.pth')
        >>> transform = tio.HistogramStandardization('path_to_landmarks.pth')
    """

    def __init__(self, landmarks: TypeLandmarks, masking_method: TypeMaskingMethod=None, **kwargs):
        super().__init__(masking_method=masking_method, **kwargs)
        self.landmarks = landmarks
        self.landmarks_dict = self._parse_landmarks(landmarks)
        self.args_names = ['landmarks', 'masking_method']

    @staticmethod
    def _parse_landmarks(landmarks: TypeLandmarks) -> dict[str, np.ndarray]:
        if isinstance(landmarks, (str, Path)):
            path = Path(landmarks)
            if path.suffix not in ('.pt', '.pth'):
                message = f'The landmarks file must have extension .pt or .pth, not "{path.suffix}"'
                raise ValueError(message)
            landmarks_dict = torch.load(path)
        else:
            landmarks_dict = landmarks
        for key, value in landmarks_dict.items():
            if isinstance(value, (str, Path)):
                landmarks_dict[key] = np.load(value)
        return landmarks_dict

    def apply_normalization(self, subject: Subject, image_name: str, mask: torch.Tensor) -> None:
        if image_name not in self.landmarks_dict:
            keys = tuple(self.landmarks_dict.keys())
            message = f'Image name "{image_name}" should be a key in the landmarks dictionary, whose keys are {keys}'
            raise KeyError(message)
        image = subject[image_name]
        landmarks = self.landmarks_dict[image_name]
        normalized = _normalize(image.data, landmarks, mask=mask.numpy())
        image.set_data(normalized)

    @classmethod
    def train(cls, images_paths: Sequence[TypePath], cutoff: tuple[float, float] | None=None, mask_path: Sequence[TypePath] | TypePath | None=None, masking_function: Callable | None=None, output_path: TypePath | None=None, *, progress: bool=True) -> np.ndarray:
        """Extract average histogram landmarks from images used for training.

        Args:
            images_paths: List of image paths used to train.
            cutoff: Optional minimum and maximum quantile values,
                respectively, that are used to select a range of intensity of
                interest. Equivalent to :math:`pc_1` and :math:`pc_2` in
                `Nyúl and Udupa's paper <https://pubmed.ncbi.nlm.nih.gov/10571928/>`_.
            mask_path: Path (or list of paths) to a binary image that will be
                used to select the voxels use to compute the stats during
                histogram training. If ``None``, all voxels in the image will
                be used.
            masking_function: Function used to extract voxels used for
                histogram training.
            output_path: Optional file path with extension ``.txt`` or
                ``.npy``, where the landmarks will be saved.

        Example:

            >>> import torch
            >>> import numpy as np
            >>> from pathlib import Path
            >>> from torchio.transforms import HistogramStandardization
            >>>
            >>> t1_paths = ['subject_a_t1.nii', 'subject_b_t1.nii.gz']
            >>> t2_paths = ['subject_a_t2.nii', 'subject_b_t2.nii.gz']
            >>>
            >>> t1_landmarks_path = Path('t1_landmarks.npy')
            >>> t2_landmarks_path = Path('t2_landmarks.npy')
            >>>
            >>> t1_landmarks = (
            ...     t1_landmarks_path
            ...     if t1_landmarks_path.is_file()
            ...     else HistogramStandardization.train(t1_paths)
            ... )
            >>> np.save(t1_landmarks_path, t1_landmarks)
            >>>
            >>> t2_landmarks = (
            ...     t2_landmarks_path
            ...     if t2_landmarks_path.is_file()
            ...     else HistogramStandardization.train(t2_paths)
            ... )
            >>> np.save(t2_landmarks_path, t2_landmarks)
            >>>
            >>> landmarks_dict = {
            ...     't1': t1_landmarks,
            ...     't2': t2_landmarks,
            ... }
            >>>
            >>> transform = HistogramStandardization(landmarks_dict)
        """
        is_masks_list = isinstance(mask_path, Sequence)
        if is_masks_list and len(mask_path) != len(images_paths):
            message = f'Different number of images ({len(images_paths)}) and mask ({len(mask_path)}) paths found'
            raise ValueError(message)
        quantiles_cutoff = DEFAULT_CUTOFF if cutoff is None else cutoff
        percentiles_cutoff = 100 * np.array(quantiles_cutoff)
        percentiles_database = []
        a, b = percentiles_cutoff
        percentiles = _get_percentiles((a, b))
        iterable: Iterable[TypePath]
        iterable = tqdm(images_paths) if progress else images_paths
        for i, image_file_path in enumerate(iterable):
            tensor, _ = read_image(image_file_path)
            if masking_function is not None:
                mask = masking_function(tensor)
            elif mask_path is None:
                mask = np.ones_like(tensor, dtype=bool)
            else:
                if is_masks_list:
                    assert isinstance(mask_path, Sequence)
                    path = mask_path[i]
                else:
                    path = mask_path
                mask, _ = read_image(path)
                mask = mask.numpy() > 0
            array = tensor.numpy()
            percentile_values = np.percentile(array[mask], percentiles)
            percentiles_database.append(percentile_values)
        percentiles_database_array = np.vstack(percentiles_database)
        mapping = _get_average_mapping(percentiles_database_array)
        if output_path is not None:
            output_path = Path(output_path).expanduser()
            extension = output_path.suffix
            if extension == '.txt':
                modality = 'image'
                text = f'{modality} {' '.join(map(str, mapping))}'
                output_path.write_text(text)
            elif extension == '.npy':
                np.save(output_path, mapping)
        return mapping

def read_image(path: TypePath) -> TypeDataAffine:
    try:
        result = _read_sitk(path)
    except RuntimeError as e:
        message = f'Error loading image with SimpleITK:\n{e}\n\nTrying NiBabel...'
        warnings.warn(message, stacklevel=2)
        try:
            result = _read_nibabel(path)
        except ImageFileError as e:
            message = f'File "{path}" not understood. Check supported formats by at https://simpleitk.readthedocs.io/en/master/IO.html#images and https://nipy.org/nibabel/api.html#file-formats'
            raise RuntimeError(message) from e
    return result

