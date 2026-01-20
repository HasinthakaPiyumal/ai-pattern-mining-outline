# Cluster 11

class Affine(SpatialTransform):
    """Apply affine transformation.

    Args:
        scales: Tuple :math:`(s_1, s_2, s_3)` defining the
            scaling values along each dimension.
        degrees: Tuple :math:`(\\theta_1, \\theta_2, \\theta_3)` defining the
            rotation around each axis.
        translation: Tuple :math:`(t_1, t_2, t_3)` defining the
            translation in mm along each axis.
        center: If ``'image'``, rotations and scaling will be performed around
            the image center. If ``'origin'``, rotations and scaling will be
            performed around the origin in world coordinates.
        default_pad_value: As the image is rotated, some values near the
            borders will be undefined.
            If ``'minimum'``, the fill value will be the image minimum.
            If ``'mean'``, the fill value is the mean of the border values.
            If ``'otsu'``, the fill value is the mean of the values at the
            border that lie under an
            `Otsu threshold <https://ieeexplore.ieee.org/document/4310076>`_.
            If it is a number, that value will be used.
            This parameter applies to intensity images only.
        default_pad_label: As the label map is rotated, some values near the
            borders will be undefined. This numeric value will be used to fill
            those undefined regions. This parameter applies to label maps only.
        image_interpolation: See :ref:`Interpolation`.
        label_interpolation: See :ref:`Interpolation`.
        check_shape: If ``True`` an error will be raised if the images are in
            different physical spaces. If ``False``, :attr:`center` should
            probably not be ``'image'`` but ``'center'``.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """

    def __init__(self, scales: TypeTripletFloat, degrees: TypeTripletFloat, translation: TypeTripletFloat, center: str='image', default_pad_value: str | float='minimum', default_pad_label: int | float=0, image_interpolation: str='linear', label_interpolation: str='nearest', check_shape: bool=True, **kwargs):
        super().__init__(**kwargs)
        self.scales = self.parse_params(scales, None, 'scales', make_ranges=False, min_constraint=0)
        self.degrees = self.parse_params(degrees, None, 'degrees', make_ranges=False)
        self.translation = self.parse_params(translation, None, 'translation', make_ranges=False)
        if center not in ('image', 'origin'):
            message = f'Center argument must be "image" or "origin", not "{center}"'
            raise ValueError(message)
        self.center = center
        self.use_image_center = center == 'image'
        self.default_pad_value = _parse_default_value(default_pad_value)
        if not isinstance(default_pad_label, (int, float)):
            message = 'default_pad_label must be a number, '
            message += f'but it is "{default_pad_label}"'
            raise ValueError(message)
        self.default_pad_label = float(default_pad_label)
        self.image_interpolation = self.parse_interpolation(image_interpolation)
        self.label_interpolation = self.parse_interpolation(label_interpolation)
        self.invert_transform = False
        self.check_shape = check_shape
        self.args_names = ['scales', 'degrees', 'translation', 'center', 'default_pad_value', 'default_pad_label', 'image_interpolation', 'label_interpolation', 'check_shape']

    @staticmethod
    def _get_scaling_transform(scaling_params: Sequence[float], center_lps: TypeTripletFloat | None=None) -> sitk.ScaleTransform:
        transform = sitk.ScaleTransform(3)
        scaling_params_array = np.array(scaling_params).astype(float)
        transform.SetScale(scaling_params_array)
        if center_lps is not None:
            transform.SetCenter(center_lps)
        return transform

    @staticmethod
    def _get_rotation_transform(degrees: Sequence[float], translation: Sequence[float], center_lps: TypeTripletFloat | None=None) -> sitk.Euler3DTransform:

        def ras_to_lps(triplet: Sequence[float]):
            return np.array((-1, -1, 1), dtype=float) * np.asarray(triplet)
        transform = sitk.Euler3DTransform()
        radians = np.radians(degrees).tolist()
        radians_lps = ras_to_lps(radians)
        translation_lps = ras_to_lps(translation)
        transform.SetRotation(*radians_lps)
        transform.SetTranslation(translation_lps)
        if center_lps is not None:
            transform.SetCenter(center_lps)
        return transform

    def get_affine_transform(self, image):
        scaling = np.asarray(self.scales).copy()
        rotation = np.asarray(self.degrees).copy()
        translation = np.asarray(self.translation).copy()
        if image.is_2d():
            scaling[2] = 1
            rotation[:-1] = 0
        if self.use_image_center:
            center_lps = image.get_center(lps=True)
        else:
            center_lps = None
        scaling_transform = self._get_scaling_transform(scaling, center_lps=center_lps)
        rotation_transform = self._get_rotation_transform(rotation, translation, center_lps=center_lps)
        sitk_major_version = get_major_sitk_version()
        if sitk_major_version == 1:
            transform = sitk.Transform(3, sitk.sitkComposite)
            transform.AddTransform(scaling_transform)
            transform.AddTransform(rotation_transform)
        elif sitk_major_version == 2:
            transforms = [scaling_transform, rotation_transform]
            transform = sitk.CompositeTransform(transforms)
        transform = transform.GetInverse()
        if self.invert_transform:
            transform = transform.GetInverse()
        return transform

    def get_default_pad_value(self, tensor: torch.Tensor, sitk_image: sitk.Image) -> float:
        default_value: float
        if self.default_pad_value == 'minimum':
            default_value = tensor.min().item()
        elif self.default_pad_value == 'mean':
            default_value = get_borders_mean(sitk_image, filter_otsu=False)
        elif self.default_pad_value == 'otsu':
            default_value = get_borders_mean(sitk_image, filter_otsu=True)
        else:
            assert isinstance(self.default_pad_value, Number)
            default_value = float(self.default_pad_value)
        return default_value

    def apply_transform(self, subject: Subject) -> Subject:
        if self.check_shape:
            subject.check_consistent_spatial_shape()
        default_value: float
        for image in self.get_images(subject):
            transform = self.get_affine_transform(image)
            transformed_tensors = []
            for tensor in image.data:
                sitk_image = nib_to_sitk(tensor[np.newaxis], image.affine, force_3d=True)
                if image[TYPE] != INTENSITY:
                    interpolation = self.label_interpolation
                    default_value = self.default_pad_label
                else:
                    interpolation = self.image_interpolation
                    default_value = self.get_default_pad_value(tensor, sitk_image)
                transformed_tensor = self.apply_affine_transform(sitk_image, transform, interpolation, default_value)
                transformed_tensors.append(transformed_tensor)
            image.set_data(torch.stack(transformed_tensors))
        return subject

    def apply_affine_transform(self, sitk_image: sitk.Image, transform: sitk.Transform, interpolation: str, default_value: float) -> torch.Tensor:
        floating = reference = sitk_image
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(self.get_sitk_interpolator(interpolation))
        resampler.SetReferenceImage(reference)
        resampler.SetDefaultPixelValue(float(default_value))
        resampler.SetOutputPixelType(sitk.sitkFloat32)
        resampler.SetTransform(transform)
        resampled = resampler.Execute(floating)
        np_array = sitk.GetArrayFromImage(resampled)
        np_array = np_array.transpose()
        tensor = torch.as_tensor(np_array)
        return tensor

def get_major_sitk_version() -> int:
    version = getattr(sitk, '__version__', None)
    major_version = 1 if version is None else 2
    return major_version

