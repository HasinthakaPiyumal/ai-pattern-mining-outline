# Cluster 16

@app.command()
def main(input_path: Path=typer.Argument(..., exists=True, file_okay=True, dir_okay=True, readable=True), transform_name: str=typer.Argument(...), output_path: Path=typer.Argument(..., file_okay=True, dir_okay=False, writable=True), kwargs: str=typer.Option(None, '--kwargs', '-k', help='String of kwargs, e.g. "degrees=(-5,15) num_transforms=3".'), imclass: str=typer.Option('ScalarImage', '--imclass', '-c', help='Name of the subclass of torchio.Image that will be used to instantiate the image.'), seed: int=typer.Option(None, '--seed', '-s', help='Seed for PyTorch random number generator.'), verbose: bool=typer.Option(False, help='Print random transform parameters.'), show_progress: bool=typer.Option(True, '--show-progress/--hide-progress', '-p/-P', help='Show animations indicating progress.')):
    """Apply transform to an image.

    Example:
    $ tiotr input.nrrd RandomMotion output.nii "degrees=(-5,15) num_transforms=3" -v
    """
    import torch
    import torchio.transforms as transforms
    from torchio.utils import apply_transform_to_file
    try:
        transform_class = getattr(transforms, transform_name)
    except AttributeError as error:
        message = f'Transform "{transform_name}" not found in torchio'
        raise ValueError(message) from error
    params_dict = get_params_dict_from_kwargs(kwargs)
    transform = transform_class(**params_dict)
    if seed is not None:
        torch.manual_seed(seed)
    with Progress(SpinnerColumn(), TextColumn('[progress.description]{task.description}'), transient=True, disable=not show_progress) as progress:
        progress.add_task('Applying transform', total=1)
        apply_transform_to_file(input_path, transform, output_path, verbose=verbose, class_=imclass)

def apply_transform_to_file(input_path: TypePath, transform, output_path: TypePath, class_: str='ScalarImage', verbose: bool=False):
    from . import data
    image = getattr(data, class_)(input_path)
    subject = data.Subject(image=image)
    transformed = transform(subject)
    transformed.image.save(output_path)
    if verbose and transformed.history:
        print('Applied transform:', transformed.history[0])

