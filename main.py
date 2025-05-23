import os
import logging
import pathlib

import click

from config import load_plot_config
from process import compute_brightness_data, generate_plots as plot_processing_results
from optimize import plot_optimization_results, fit_models


@click.command()
@click.option("--output_path", "-o", default=None, help="Output directory")
@click.option(
    "--output_postfix",
    "-p",
    default="",
    help="Postfix for output directory, works if output_path is None."
    + "Outputs will be saved to ${video_directory}/${output_postfix}",
)
@click.option("--config_path", "-c", default="config.yaml", help="Config file path")
@click.option("--verbose", "-v", count=True, default=0, help="Increase verbosity")
@click.option("--quiet", "-q", count=True, default=0, help="Decrease verbosity")
@click.argument("video_path", type=click.Path(exists=True))
def main(
    video_path: str,
    output_postfix: str,
    config_path: str,
    verbose: int,
    quiet: int,
    output_path: int | None,
):
    verbosity = verbose - quiet
    logging.basicConfig(
        level=(logging.INFO - verbosity * 10),
        format="[%(asctime)s] [%(levelname)-5s]: %(message)s",
        datefmt="%Y-%m-%d] [%H:%M:%S",
        force=True,
    )
    logger = logging.getLogger(__name__)
    if output_path is not None:
        output_dir = pathlib.Path(output_path)
    else:
        output_dir = pathlib.Path(os.path.dirname(video_path)) / output_postfix

    plot_config = load_plot_config(config_path)
    data = compute_brightness_data(video_path, output_dir)
    plot_processing_results(data, plot_config, output_dir)
    results = fit_models(data["time"], data["difference"], plot_config)
    plot_optimization_results(
        results, data["time"], data["difference"], plot_config, output_dir
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter (wrapped by click)
