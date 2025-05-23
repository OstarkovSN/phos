import os
import logging
import yaml

logger = logging.getLogger(__name__)


def load_plot_config(config_path):
    """Load plot configuration from YAML file, creating default if missing."""
    try:
        logger.debug(f"Loading plot configuration from {config_path}")
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        # Add missing sections
        default_additions = {
            "exponential": {"title": "Exponential Decay Fit", "color": "blue"},
            "hyperbolic": {"title": "Hyperbolic Decay Fit", "color": "green"},
            "generalized_hyperbolic": {
                "title": "Generalized Hyperbolic Decay Fit",
                "color": "purple",
            },
        }
        updated = False
        for key, value in default_additions.items():
            if key not in config:
                config[key] = value
                updated = True
        if updated:
            with open(config_path, "w") as file:
                yaml.dump(config, file, default_flow_style=False)
        return config
    except FileNotFoundError:
        logger.info(
            f"Config file not found at {config_path}. Creating default config..."
        )
        default_config = {
            "background": {
                "title": "Background Brightness Over Time",
                "points": {
                    "name": "Background Brightness Over Time",
                    "mode": "markers",
                    "marker": {"color": "blue", "opacity": 0.7},
                },
            },
            "object": {
                "title": "Object Brightness Over Time",
                "points": {
                    "name": "Object Brightness Over Time",
                    "mode": "markers",
                    "marker": {"color": "blue", "opacity": 0.7},
                },
            },
            "difference": {
                "title": "Object - Background Brightness Difference",
                "points": {
                    "name": "Object - Background Brightness Difference",
                    "mode": "markers",
                    "marker": {"color": "blue", "opacity": 0.7},
                },
            },
            "exponential": {
                "title": "Exponential Decay Fit",
                "fit": {
                    "name": "Exponential Decay Fit",
                    "mode": "lines",
                    "line": {"color": "blue"},
                },
                "points": {
                    "name": "Experiment",
                    "mode": "markers",
                    "marker": {"color": "blue", "opacity": 0.1},
                },
                "bounds": [[0, 0], ["inf", "inf"]],
            },
            "hyperbolic": {
                "title": "Hyperbolic Decay Fit",
                "fit": {
                    "name": "Hyperbolic Decay Fit",
                    "mode": "lines",
                    "line": {"color": "blue"},
                },
                "points": {
                    "name": "Experiment",
                    "mode": "markers",
                    "marker": {"color": "blue", "opacity": 0.1},
                },
                "bounds": [[0, 0], ["inf", "inf"]],
            },
            "generalized_hyperbolic": {
                "title": "Generalized Hyperbolic Decay Fit",
                "fit": {
                    "name": "Generalized Hyperbolic Decay Fit",
                    "mode": "lines",
                    "line": {"color": "blue"},
                },
                "points": {
                    "name": "Experiment",
                    "mode": "markers",
                    "marker": {"color": "blue", "opacity": 0.1},
                },
                "bounds": [[0, 0, 1], ["inf", "inf", 2]],
                "p0": [1, 1, 1],
            },
            "defaults": {"width": 1200, "height": 600, "scale": 2},
        }
        if os.path.dirname(config_path):
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as file:
            yaml.dump(default_config, file, default_flow_style=False)
        return default_config
    except PermissionError:
        logger.error("No write permission for config file")
        raise
