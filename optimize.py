import os
import logging
from typing import Dict, Callable, Any, Optional, List, Tuple
import pathlib

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import plotly.io as pio

# Set default image format
pio.kaleido.scope.default_format = "png"

logger = logging.getLogger(__name__)


def exponential_model(t: np.ndarray, I0: float, k: float) -> np.ndarray:
    """Exponential decay: I(t) = I0 * exp(-k*t)"""
    return I0 * np.exp(-k * t)


def hyperbolic_model(t: np.ndarray, I0: float, K: float) -> np.ndarray:
    """Hyperbolic decay: I(t) = I0 / (1 + K*t + 1e-8)^2"""
    return I0 / (1 + K * t + 1e-8) ** 2


def generalized_hyperbolic_model(
    t: np.ndarray, I0: float, M: float, p: float
) -> np.ndarray:
    """Generalized hyperbolic decay: I(t) = I0 / (1 + M*t + 1e-8)^clamp(p,1,2)"""
    return I0 / (1 + M * t + 1e-8) ** np.clip(p, 1, 2)


def fit_models(
    time: np.ndarray,
    intensity: np.ndarray,
    config: Dict[str, Any],
    models: Optional[Dict[str, Callable[..., np.ndarray]]] = None,
    param_names: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Fit models to intensity data using non-linear regression."""

    if models is None:
        models = {
            "exponential": exponential_model,
            "hyperbolic": hyperbolic_model,
            "generalized_hyperbolic": generalized_hyperbolic_model,
        }

        param_names = {
            "exponential": ["I0", "k"],
            "hyperbolic": ["I0", "K"],
            "generalized_hyperbolic": ["I0", "M", "p"],
        }
    if param_names is None:
        param_names = {
            model_name: model.__code__.co_varnames[1:]
            for model_name, model in models.items()
        }

    results: Dict[str, Dict[str, Any]] = {}
    time = np.asarray(time)
    intensity = np.asarray(intensity)

    if np.any(np.isnan(intensity)) or np.any(np.isinf(intensity)):
        logger.error("Intensity data contains NaN/Inf values")
        return results

    for model_name, model_func in models.items():
        try:
            logger.info(f"Fitting {model_name} model...")
            bounds = list(
                map(
                    lambda tup: tuple(map(float, tup)),
                    config.get(model_name, {}).get(
                        "bounds",
                        (
                            tuple(-np.inf for _ in range(len(param_names[model_name]))),
                            tuple(np.inf for _ in range(len(param_names[model_name]))),
                        ),
                    ),
                )
            )
            p0 = list(config[model_name]["p0"]) if model_name in config else None

            maxfev: int = config.get(model_name, {}).get("maxfev", 5000)

            popt, _ = curve_fit(
                model_func, time, intensity, bounds=bounds, maxfev=maxfev
            )

            predicted = model_func(time, *popt)
            if np.any(np.isnan(predicted)) or np.any(np.isinf(predicted)):
                raise ValueError("Predictions contain NaN/Inf")

            r2: float = r2_score(intensity, predicted)
            results[model_name] = {
                "params": popt.tolist(),
                "r2": r2,
                "model": model_func,
                "model_name": model_name,
                "param_names": param_names[model_name],
            }
            logger.info(f"R^2 for {model_name} model: {r2}")

        except Exception as e:
            logger.error(f"Model fitting failed for {model_name}: {e}")
            results[model_name] = {
                "params": None,
                "r2": -np.inf,
                "model": model_func,
                "model_name": model_name,
                "param_names": param_names[model_name],
            }

    return results


def plot_optimization_results(
    results: Dict[str, Dict[str, Any]],
    time: np.ndarray,
    intensity: np.ndarray,
    config: Dict[str, Any],
    output_dir: str | pathlib.Path,
) -> None:
    """
    Generate and save comparison plots for all models using Plotly.

    Parameters:
        results (dict): Results from fit_models
        time (np.array): Time points
        intensity (np.array): Experimental intensity data
        config (dict): Configuration dictionary from YAML
        output_dir (str): Directory to save plots
    """
    logger.info("Generating optimization plots...")
    if not isinstance(output_dir, pathlib.Path):
        output_dir = pathlib.Path(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    defaults = config.get("defaults", {})
    width: int = defaults.get("width", 1200)
    height: int = defaults.get("height", 600)

    time_sorted = np.sort(time)

    for model_name, result in results.items():
        # Initialize Plotly figure
        fig = go.Figure()

        # Add experimental data
        fig.add_trace(
            go.Scatter(
                x=time,
                y=intensity,
                **config[model_name]["points"],
            )
        )

        # Predict based on model
        params: List[float] = result["params"]
        if params is None:
            logger.error(
                f"Model fitting failed for {model_name}. Skipping plot generation."
            )
        else:
            pred = result["model"](time_sorted, *params)
            param_str_unformatted = ", ".join(
                f"{param_name}: {param_value:.4f}"
                for param_name, param_value in zip(result["param_names"], params)
            )

            # Add model prediction
            fig.add_trace(
                go.Scatter(
                    x=time_sorted,
                    y=pred,
                    **config[model_name]["fit"],
                )
            )

            # Layout configuration
            fig.update_layout(
                title=config[model_name]["title"],
                xaxis_title="Time (s)",
                yaxis_title="Intensity",
                legend_title="Model",
                width=width,
                height=height,
                template="plotly_white",
                legend=dict(
                    yanchor="top", y=0.99, xanchor="right", x=0.99, font=dict(size=12)
                ),
                margin=dict(l=50, r=50, t=80, b=100),
                showlegend=True,
                annotations=[
                    dict(
                        x=0.5,
                        y=-0.15,
                        text=param_str_unformatted,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        xanchor="center",
                        yanchor="top",
                    ),
                    dict(
                        x=0.1,
                        y=0.9,
                        text=f"R^2: {result['r2']:.4f}",
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        xanchor="center",
                        yanchor="top",
                    ),
                ],
            )

            # Save plot
            filename = f"{model_name.lower().replace(' ', '_')}_fit.png"
            fig.write_image(output_dir / "plots" / filename)

    logger.info("Optimization plots generated successfully.")
