# Cluster 31

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

def make_video(image: ScalarImage, output_path: TypePath, seconds: float | None=None, frame_rate: float | None=None, direction: str='I', verbosity: str='error') -> None:
    ffmpeg = get_ffmpeg()
    if seconds is None and frame_rate is None:
        message = 'Either seconds or frame_rate must be provided.'
        raise ValueError(message)
    if seconds is not None and frame_rate is not None:
        message = 'Provide either seconds or frame_rate, not both.'
        raise ValueError(message)
    if image.num_channels > 1:
        message = 'Only single-channel tensors are supported for video output for now.'
        raise ValueError(message)
    tmin, tmax = (image.data.min(), image.data.max())
    if tmin < 0 or tmax > 255:
        message = 'The tensor must be in the range [0, 256) for video output. The image data will be rescaled to this range.'
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        image = RescaleIntensity((0, 255))(image)
    if image.data.dtype != torch.uint8:
        message = 'Only uint8 tensors are supported for video output. The image data will be cast to uint8.'
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        image = To(torch.uint8)(image)
    direction = direction.upper()
    if direction == 'I':
        target = 'IPL'
    elif direction == 'S':
        target = 'SPL'
    elif direction == 'A':
        target = 'AIL'
    elif direction == 'P':
        target = 'PIL'
    elif direction == 'R':
        target = 'RIP'
    elif direction == 'L':
        target = 'LIP'
    else:
        message = f'Direction must be one of "I", "S", "P", "A", "R" or "L". Got {direction!r}.'
        raise ValueError(message)
    image = ToOrientation(target)(image)
    spacing_f, spacing_h, spacing_w = image.spacing
    if spacing_h != spacing_w:
        message = f'The height and width spacings should be the same video output. Got {spacing_h:.2f} and {spacing_w:.2f}. Resampling both to {spacing_f:.2f}.'
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        spacing_iso = min(spacing_h, spacing_w)
        target_spacing = (spacing_f, spacing_iso, spacing_iso)
        image = Resample(target_spacing)(image)
    num_frames, height, width = image.spatial_shape
    if height % 2 != 0 or width % 2 != 0:
        message = f'The height ({height}) and width ({width}) must be even. The image will be cropped to the nearest even number.'
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        image = EnsureShapeMultiple((1, 2, 2), method='crop')(image)
    if seconds is not None:
        frame_rate = num_frames / seconds
    output_path = Path(output_path)
    if output_path.suffix.lower() != '.mp4':
        message = 'Only .mp4 files are supported for video output.'
        raise NotImplementedError(message)
    frames = image.numpy()[0]
    first = frames[0]
    height, width = first.shape
    process = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='gray', s=f'{width}x{height}', framerate=frame_rate).output(str(output_path), vcodec='libx265', pix_fmt='yuv420p', loglevel=verbosity, **{'x265-params': f'log-level={verbosity}'}).overwrite_output().run_async(pipe_stdin=True)
    for array in frames:
        buffer = array.tobytes()
        process.stdin.write(buffer)
    process.stdin.close()
    process.wait()

