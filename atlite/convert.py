# SPDX-FileCopyrightText: Contributors to atlite <https://github.com/pypsa/atlite>
#
# SPDX-License-Identifier: MIT
"""
All functions for converting weather data into energy data.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections import namedtuple
from collections.abc import Callable
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from dask import compute, delayed
from dask.array import absolute, arccos, cos, maximum, mod, radians, sin, sqrt
from dask.diagnostics import ProgressBar
from numpy import pi
from scipy.sparse import csr_matrix

from atlite import csp as cspm
from atlite import hydro as hydrom
from atlite import wind as windm
from atlite.aggregate import aggregate_matrix
from atlite.gis import spdiag
from atlite.pv.irradiation import TiltedIrradiation
from atlite.pv.orientation import SurfaceOrientation, get_orientation
from atlite.pv.solar_panel_model import SolarPanelModel
from atlite.pv.solar_position import SolarPosition
from atlite.resource import (
    get_cspinstallationconfig,
    get_solarpanelconfig,
    get_windturbineconfig,
    windturbine_smooth,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from typing import Literal

    from atlite.cutout import Cutout
    from atlite.resource import TurbineConfig


def convert_and_aggregate(
    cutout: Cutout,
    convert_func: Callable[..., xr.DataArray],
    matrix: xr.DataArray | csr_matrix | None = None,
    index: pd.Index | None = None,
    layout: xr.DataArray | None = None,
    shapes: list | pd.Series | gpd.GeoSeries | gpd.GeoDataFrame | None = None,
    shapes_crs: int | str | Any = 4326,
    mean_over_time: bool = False,
    sum_over_time: bool = False,
    capacity_factor: bool = False,
    return_capacity: bool = False,
    capacity_units: str = "MW",
    show_progress: bool = False,
    dask_kwargs: dict[str, Any] | None = None,
    **convert_kwds: Any,
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    """
    Convert and aggregate weather data to energy data.

    This is a gateway function called by all individual time-series
    generation functions (e.g., pv, wind). It is not intended for direct
    user interaction. All parameters are also available in the specific
    generation functions.

    Parameters
    ----------
    cutout : Cutout
        The weather data cutout containing the input data.
    convert_func : Callable[..., xr.DataArray]
        Function that converts weather data to energy data.
    matrix : xr.DataArray | csr_matrix | None, optional
        Aggregation matrix (N x S) where N is the number of buses and S is
        the number of spatial coordinates in the order of `cutout.grid`.
        Used to aggregate grid cells to buses.
    index : pd.Index | None, optional
        Index of buses for the aggregated output.
    layout : xr.DataArray | None, optional
        Capacity layout (X x Y) specifying the capacity to be built in each
        grid cell.
    shapes : list | pd.Series | gpd.GeoSeries | gpd.GeoDataFrame | None, optional
        Geometric shapes (e.g., polygons) used to construct an indicator matrix.
        The shapes' index determines the bus index in the time-series.
    shapes_crs : int | str | Any, default 4326
        Coordinate reference system for shapes. If different from the cutout's
        CRS, shapes are transformed to match (defaults to EPSG:4326).
    mean_over_time : bool, default False
        If True, the result is time-averaged.
    sum_over_time : bool, default False
        If True, the result is summed over time.
    capacity_factor : bool, default False
        If True, normalizes the result by installed capacity at each bus.
    return_capacity : bool, default False
        If True, additionally returns the installed capacity at each bus
        corresponding to the layout.
    capacity_units : str, default "MW"
        Units for installed capacity (only relevant when `capacity_factor`
        or `return_capacity` is True).
    show_progress : bool, default False
        Whether to display a progress bar during computation.
    dask_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments passed to `dask.compute()`.
    **convert_kwds : Any
        Additional keyword arguments passed to the conversion function.

    Returns
    -------
    xr.DataArray | tuple[xr.DataArray, xr.DataArray]
        If `return_capacity` is False: Single DataArray with time-series of the
        selected resource (per grid cell, aggregated per bus, or time-averaged).
        If `return_capacity` is True: Tuple of (resource_data, capacity_data).

    """
    # Handle mutable default arguments.
    if dask_kwargs is None:
        dask_kwargs = {}

    # Check whether any of matrix, shapes or layout is given. If not, no
    # aggregation is to be done.
    aggregate = any(v is not None for v in [layout, shapes, matrix])

    # Define the cases for which aggregation is possible.
    cases_with_aggregation = [
        convert_temperature,
        convert_wind,
        convert_pv,
        convert_csp,
        convert_solar_thermal,
        convert_runoff,
        convert_heat_demand,
        convert_cooling_demand,
    ]

    # Define the cases for which the installed capacity can be used.
    cases_with_capacity = [
        convert_wind,
        convert_pv,
        convert_csp,
    ]

    # Check that inputs are valid.
    if aggregate and convert_func not in cases_with_aggregation:
        raise ValueError(
            "`matrix`, `shapes` or `layout` can only be used with "
            "`temperature`, `wind`, `pv`, `csp`, `solar_thermal`, "
            "`runoff`, `heat_demand` or `cooling_demand`."
        )
    if capacity_factor or return_capacity:
        if convert_func not in cases_with_capacity:
            raise ValueError(
                "`capacity_factor` or `return_capacity` can only be used with "
                "`wind`, `pv`, or `csp`."
            )
        if not aggregate:
            raise ValueError(
                "`capacity_factor` or `return_capacity` requires at least one "
                "of `matrix`, `shapes` or `layout` to be passed."
            )
    if matrix is not None and shapes is not None:
        raise ValueError(
            "Passing matrix and shapes is ambiguous. Pass only one of them."
        )

    # Get the name of the conversion function.
    func_name = convert_func.__name__.replace("convert_", "")

    # Log the conversion function being used.
    logger.info(f"Convert and aggregate '{func_name}'.")

    # Convert the weather data to energy data using the provided function.
    da = convert_func(cutout.data, **convert_kwds)

    if aggregate:
        # Get the matrix and index for aggregation.
        matrix, index = get_matrix_and_index(
            cutout,
            matrix=matrix,
            index=index,
            layout=layout,
            shapes=shapes,
            shapes_crs=shapes_crs,
        )

        # Aggregate the converted data.
        results = aggregate_matrix(da, matrix=matrix, index=index)

        if capacity_factor or return_capacity:
            # Calculate installed capacity at each bus.
            caps = matrix.sum(-1)

            # Create a DataArray for installed capacity.
            capacity = xr.DataArray(np.asarray(caps).flatten(), [index])
            capacity.attrs["units"] = capacity_units

            if capacity_factor:
                # Divide the time-series by installed capacity.
                results = (results / capacity.where(capacity != 0)).fillna(0.0)
                results.attrs["units"] = "per unit of installed capacity"
            else:
                results.attrs["units"] = capacity_units

            # Apply time averaging or summation if requested.
            if mean_over_time:
                results = results.mean(dim="time")
            elif sum_over_time:
                results = results.sum(dim="time")

        if return_capacity:
            return maybe_progressbar(results, show_progress, **dask_kwargs), capacity
        else:
            return maybe_progressbar(results, show_progress, **dask_kwargs)

    else:
        # Apply time averaging or summation if requested.
        if mean_over_time:
            results = da.mean(dim="time")
        elif sum_over_time:
            results = da.sum(dim="time")
        else:
            results = da

        return maybe_progressbar(results, show_progress, **dask_kwargs)


def get_matrix_and_index(
    cutout: Cutout,
    matrix: xr.DataArray | csr_matrix | None = None,
    index: pd.Index | None = None,
    layout: xr.DataArray | None = None,
    shapes: list | pd.Series | gpd.GeoSeries | gpd.GeoDataFrame | None = None,
    shapes_crs: int | str | Any = 4326,
) -> tuple[csr_matrix, pd.Index]:
    """
    Construct the aggregation matrix and index for spatial aggregation.

    Parameters
    ----------
    cutout : Cutout
        The weather data cutout containing spatial grid information.
    matrix : xr.DataArray | csr_matrix | None, optional
        Pre-defined aggregation matrix (N x S) where N is the number of buses
        and S is the number of spatial coordinates in `cutout.grid` order.
    index : pd.Index | None, optional
        Index labels for the buses in the aggregated output.
    layout : xr.DataArray | None, optional
        Capacity layout (X x Y) specifying the capacity to be built in each
        grid cell.
    shapes : list | pd.Series | gpd.GeoSeries | gpd.GeoDataFrame | None, optional
        Geometric shapes used to construct an indicator matrix. The shapes'
        index determines the bus labels in the time-series.
    shapes_crs : int | str | Any, default 4326
        Coordinate reference system for shapes. If different from the cutout's
        CRS, shapes are transformed to match (defaults to EPSG:4326).

    Returns
    -------
    tuple[csr_matrix, pd.Index]
        A tuple containing:
        - Sparse aggregation matrix for spatial aggregation
        - Index labels for the aggregated buses
    """
    if matrix is not None:
        if isinstance(matrix, xr.DataArray):
            # Check that the spatial coordinates of the matrix align with
            # the cutout spatial coordinates.
            coords = matrix.indexes.get(matrix.dims[1]).to_frame(index=False)
            if not np.array_equal(coords[["x", "y"]], cutout.grid[["x", "y"]]):
                raise ValueError(
                    "Matrix spatial coordinates not aligned with cutout spatial "
                    "coordinates."
                )

            if index is None:
                index = matrix

        if not matrix.ndim == 2:
            raise ValueError("Matrix not 2-dimensional.")

        matrix = csr_matrix(matrix)

    if shapes is not None:
        # If shapes are given as a GeoDataFrame or GeoSeries, extract the
        # index.
        geoseries_like = (pd.Series, gpd.GeoDataFrame, gpd.GeoSeries)
        if isinstance(shapes, geoseries_like) and index is None:
            index = shapes.index

        # Construct indicator matrix from shapes.
        matrix = cutout.indicatormatrix(shapes, shapes_crs)

    if layout is not None:
        # Check that layout is an xarray DataArray.
        assert isinstance(layout, xr.DataArray)

        # Reindex and stack layout to match cutout grid.
        layout = layout.reindex_like(cutout.data).stack(spatial=["y", "x"])

        # Construct the layout matrix.
        if matrix is None:
            matrix = csr_matrix(layout.expand_dims("new"))
        else:
            matrix = csr_matrix(matrix) * spdiag(layout)

    # If there is still no index, create a default one.
    if index is None:
        index = pd.RangeIndex(matrix.shape[0])

    return matrix, index


def maybe_progressbar(
    ds: xr.DataArray, show_progress: bool, **kwargs: Any
) -> xr.DataArray:
    """
    Load a xr.dataset with dask arrays either with or without progressbar.

    Parameters
    ----------
    ds : xr.DataArray
        The DataArray to load.
    show_progress : bool
        Whether to show a progress bar.
    **kwargs : Any
        Additional keyword arguments passed to ds.load().

    Returns
    -------
    xr.DataArray
        The loaded DataArray.
    """
    if show_progress:
        with ProgressBar(minimum=2):
            ds.load(**kwargs)
    else:
        ds.load(**kwargs)
    return ds


# temperature
def convert_temperature(ds: xr.Dataset) -> xr.DataArray:
    """
    Convert temperature data from Kelvin to degree Celsius.

    Supports multiple temperature variable names including 'temperature',
    'soil temperature', and 'dewpoint temperature'. For soil temperature,
    NaN values over sea areas are set to zero.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing temperature data in Kelvin.

    Returns
    -------
    xr.DataArray
        Temperature data converted to degree Celsius with appropriate
        variable name and units attributes.

    Raises
    ------
    ValueError
        If none of the expected temperature variables are found in the dataset.
    """
    # Define possible variable names for temperature.
    variable_names = ["temperature", "soil temperature", "dewpoint temperature"]

    # Check that at least one variable name is in the dataset.
    if not any(name in ds for name in variable_names):
        raise ValueError(
            f"None of the temperature variables {variable_names} found in dataset."
        )

    # Get the variable name that is in the dataset.
    variable_name = next(name for name in variable_names if name in ds)

    # Convert temperature from Kelvin to degree Celsius.
    ds = ds[variable_name] - 273.15

    # For soil temperature, there are NaNs where there is sea; set them to zero.
    ds = ds.fillna(0.0)

    # Set name and units attribute.
    ds = ds.rename(variable_name)
    ds.attrs["units"] = "degree Celsius"

    return ds


def temperature(
    cutout: Cutout, **params: Any
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    """
    Generate temperature time-series from weather data.

    Converts temperature data from Kelvin to Celsius and optionally
    aggregates spatially and/or temporally based on the provided parameters.

    Parameters
    ----------
    cutout : Cutout
        The weather data cutout containing temperature data.
    **params : Any
        Additional parameters for spatial/temporal aggregation. See
        `convert_and_aggregate` for available options.

    Returns
    -------
    xr.DataArray | tuple[xr.DataArray, xr.DataArray]
        Temperature time-series in degree Celsius. If `return_capacity` is True,
        returns a tuple of (temperature_data, capacity_data).
    """
    return cutout.convert_and_aggregate(convert_func=convert_temperature, **params)


def soil_temperature(
    cutout: Cutout, **params: Any
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    """
    Generate soil temperature time-series from weather data.

    Converts soil temperature data from Kelvin to Celsius. NaN values
    over sea areas are automatically set to zero.

    Parameters
    ----------
    cutout : Cutout
        The weather data cutout containing soil temperature data.
    **params : Any
        Additional parameters for spatial/temporal aggregation. See
        `convert_and_aggregate` for available options.

    Returns
    -------
    xr.DataArray | tuple[xr.DataArray, xr.DataArray]
        Soil temperature time-series in degree Celsius. If `return_capacity` is True,
        returns a tuple of (temperature_data, capacity_data).
    """
    return cutout.convert_and_aggregate(convert_func=convert_temperature, **params)


def dewpoint_temperature(
    cutout: Cutout, **params: Any
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    """
    Generate dewpoint temperature time-series from weather data.

    Converts dewpoint temperature data from Kelvin to Celsius.

    Parameters
    ----------
    cutout : Cutout
        The weather data cutout containing dewpoint temperature data.
    **params : Any
        Additional parameters for spatial/temporal aggregation. See
        `convert_and_aggregate` for available options.

    Returns
    -------
    xr.DataArray | tuple[xr.DataArray, xr.DataArray]
        Dewpoint temperature time-series in degree Celsius. If `return_capacity` is True,
        returns a tuple of (temperature_data, capacity_data).
    """
    return cutout.convert_and_aggregate(convert_func=convert_temperature, **params)


def convert_coefficient_of_performance(
    ds: xr.Dataset,
    source: str,
    sink_T: float,
    c0: float | None,
    c1: float | None,
    c2: float | None,
) -> xr.DataArray:
    """
    Convert temperature data to coefficient of performance for heat pumps.

    Calculate the coefficient of performance (COP) for heat pumps based on
    temperature differences between source and sink using a quadratic model.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing temperature data.
    source : str
        Heat source type, either 'air' or 'soil'.
    sink_T : float
        Sink temperature in degree Celsius.
    c0 : float or None
        Quadratic model coefficient (intercept). If None, uses defaults:
        6.81 for air source, 8.77 for soil source.
    c1 : float or None
        Quadratic model coefficient (linear term). If None, uses defaults:
        -0.121 for air source, -0.150 for soil source.
    c2 : float or None
        Quadratic model coefficient (quadratic term). If None, uses defaults:
        0.000630 for air source, 0.000734 for soil source.

    Returns
    -------
    xr.DataArray
        Coefficient of performance values with time and spatial dimensions.

    Raises
    ------
    ValueError
        If source is not 'air' or 'soil'.

    Notes
    -----
    The COP is computed using: COP = c0 + c1 * ΔT + c2 * ΔT²
    where ΔT is the temperature difference between sink and source.
    """
    if source not in ["air", "soil"]:
        raise ValueError("'source' must be one of ['air', 'soil']")

    # Get the source temperature and set default coefficients if not provided.
    source_T = convert_temperature(ds)

    match source:
        case "air":
            if c0 is None:
                c0 = 6.81
            if c1 is None:
                c1 = -0.121
            if c2 is None:
                c2 = 0.000630
        case "soil":
            if c0 is None:
                c0 = 8.77
            if c1 is None:
                c1 = -0.150
            if c2 is None:
                c2 = 0.000734

    # Calculate the temperature difference.
    delta_T = sink_T - source_T

    # Calculate the coefficient of performance.
    cop = c0 + c1 * delta_T + c2 * delta_T**2

    # Set name and units attribute.
    cop = cop.rename("coefficient of performance")
    cop.attrs["units"] = "heating or cooling delivered per unit of energy input"

    return cop


def coefficient_of_performance(
    cutout: Cutout,
    source: str = "air",
    sink_T: float = 55.0,
    c0: float | None = None,
    c1: float | None = None,
    c2: float | None = None,
    **params: Any,
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    """
    Convert ambient or soil temperature to coefficient of performance (COP) of
    air- or ground-sourced heat pumps. The COP is a function of temperature
    difference from source to sink. The defaults for either source (c0, c1, c2)
    are based on a quadratic regression in [1].

    Parameters
    ----------
    cutout : Cutout
        The weather data cutout.
    source : str, default 'air'
        The heat source. Can be 'air' or 'soil'.
    sink_T : float, default 55.0
        The temperature of the heat sink.
    c0 : float, optional
        The constant regression coefficient for the temperature difference.
    c1 : float, optional
        The linear regression coefficient for the temperature difference.
    c2 : float, optional
        The quadratic regression coefficient for the temperature difference.
    **params : Any
        Additional parameters for convert_and_aggregate.

    Returns
    -------
    xr.DataArray | tuple[xr.DataArray, xr.DataArray]
        Coefficient of performance time series and optionally capacity.

    References
    ----------
    [1] Staffell, Brett, Brandon, Hawkes, A review of domestic heat pumps,
    Energy & Environmental Science (2012), 5, 9291-9306,
    https://doi.org/10.1039/C2EE22653G.
    """
    return cutout.convert_and_aggregate(
        convert_func=convert_coefficient_of_performance,
        source=source,
        sink_T=sink_T,
        c0=c0,
        c1=c1,
        c2=c2,
        **params,
    )


# heat demand
def convert_heat_demand(
    ds: xr.Dataset, threshold: float, a: float, constant: float, hour_shift: float
) -> xr.DataArray:
    """
    Convert temperature data to heat demand using degree-day method.

    Calculate heating degree days based on daily average temperatures
    and a temperature threshold. Values are computed using the degree-day
    approximation for building heat demand.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing temperature data.
    threshold : float
        Base temperature threshold in degree Celsius below which heating
        is required.
    a : float
        Scaling factor relating temperature difference to heat demand.
    constant : float
        Constant baseline heat demand.
    hour_shift : float
        Time shift in hours to account for local time zones.

    Returns
    -------
    xr.DataArray
        Daily heating degree days with 'time' resampled to daily frequency.

    Notes
    -----
    Heat demand is calculated as: demand = constant + a * max(0, threshold - T_daily)
    where T_daily is the daily average temperature.
    """
    # Convert temperature to degree Celsius.
    temperature = convert_temperature(ds)

    # Apply a time shift to account for local time zones.
    temperature = temperature.assign_coords(
        time=(
            temperature.coords["time"] + np.timedelta64(dt.timedelta(hours=hour_shift))
        )
    )

    # Take the daily average temperature.
    temperature = temperature.resample(time="1D").mean(dim="time")

    # Calculate heat demand using degree-day approximation.
    heat_demand = constant + a * (threshold - temperature).clip(min=0.0)

    # Set name and units attribute.
    heat_demand = heat_demand.rename("heating degrees")
    heat_demand.attrs["units"] = "degree Celsius"

    return heat_demand


def heat_demand(
    cutout: Cutout,
    threshold: float = 15.0,
    a: float = 1.0,
    constant: float = 0.0,
    hour_shift: float = 0.0,
    **params: Any,
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    """
    Generate daily heat demand time-series using the degree-day approximation.

    Converts outside temperature into daily heat demand based on a linear
    relationship with temperature deviations below a threshold. Supports
    time zone adjustments for proper daily averaging.

    Parameters
    ----------
    cutout : Cutout
        The weather data cutout containing temperature data.
    threshold : float, default 15.0
        Temperature threshold in degrees Celsius above which there is no
        heat demand.
    a : float, default 1.0
        Linear factor relating heat demand to temperature difference below
        the threshold.
    constant : float, default 0.0
        Constant heat demand component independent of outside temperature
        (e.g., for water heating).
    hour_shift : float, default 0.0
        Time shift in hours relative to UTC for daily averaging.
        Examples: Moscow winter = 4, New York winter = -5.
    **params : Any
        Additional parameters for spatial/temporal aggregation. See
        `convert_and_aggregate` for available options.

    Returns
    -------
    xr.DataArray | tuple[xr.DataArray, xr.DataArray]
        Daily heat demand time-series in heating degree-days. If `return_capacity`
        is True, returns a tuple of (heat_demand_data, capacity_data).

    Notes
    -----
    The time shift applies uniformly across the entire spatial domain.
    More fine-grained, space- and time-dependent time zone control will
    be implemented in future versions.

    Warnings
    --------
    When using time shifts with monthly data, boundary effects may occur,
    potentially creating duplicate daily indices. Manual post-processing
    may be required to handle these edge cases.

    """
    return cutout.convert_and_aggregate(
        convert_func=convert_heat_demand,
        threshold=threshold,
        a=a,
        constant=constant,
        hour_shift=hour_shift,
        **params,
    )


# cooling demand
def convert_cooling_demand(
    ds: xr.Dataset, threshold: float, a: float, constant: float, hour_shift: float
) -> xr.DataArray:
    """
    Convert temperature data to cooling demand using degree-day method.

    Calculate cooling degree days based on daily average temperatures
    and a temperature threshold. Values are computed using the degree-day
    approximation for building cooling demand.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing temperature data.
    threshold : float
        Base temperature threshold in degree Celsius above which cooling
        is required.
    a : float
        Scaling factor relating temperature difference to cooling demand.
    constant : float
        Constant baseline cooling demand.
    hour_shift : float
        Time shift in hours to account for local time zones.

    Returns
    -------
    xr.DataArray
        Daily cooling degree days with 'time' resampled to daily frequency.

    Notes
    -----
    Cooling demand is calculated as: demand = constant + a * max(0, T_daily - threshold)
    where T_daily is the daily average temperature.
    """
    # Convert temperature to degree Celsius.
    temperature = convert_temperature(ds)

    # Apply a time shift to account for local time zones.
    temperature = temperature.assign_coords(
        time=(
            temperature.coords["time"] + np.timedelta64(dt.timedelta(hours=hour_shift))
        )
    )

    # Take the daily average temperature.
    temperature = temperature.resample(time="1D").mean(dim="time")

    # Calculate cooling demand using degree-day approximation.
    cooling_demand = constant + a * (temperature - threshold).clip(min=0.0)

    # Set name and units attribute.
    cooling_demand = cooling_demand.rename("cooling degrees")
    cooling_demand.attrs["units"] = "degree Celsius"

    return cooling_demand


def cooling_demand(
    cutout: Cutout,
    threshold: float = 23.0,
    a: float = 1.0,
    constant: float = 0.0,
    hour_shift: float = 0.0,
    **params: Any,
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    """
    Convert outside temperature into daily cooling demand using degree-day approximation.

    Generate daily cooling demand time-series based on temperature thresholds
    and degree-day calculations with optional time zone adjustments.

    Parameters
    ----------
    cutout : Cutout
        Weather data cutout containing temperature data.
    threshold : float, default 23.0
        Outside temperature threshold in degrees Celsius above which cooling
        demand occurs. The default 23°C follows European computational practices
        (UK Met Office and European Commission use 22°C and 24°C respectively).
    a : float, default 1.0
        Linear factor relating cooling demand to temperature difference above
        the threshold.
    constant : float, default 0.0
        Constant cooling demand component independent of outside temperature
        (e.g., for ventilation requirements).
    hour_shift : float, default 0.0
        Time shift in hours relative to UTC for daily averaging.
        Examples: Moscow summer = 3, New York summer = -4.
    **params : Any
        Additional parameters for spatial/temporal aggregation. See
        `convert_and_aggregate` for available options.

    Returns
    -------
    xr.DataArray | tuple[xr.DataArray, xr.DataArray]
        Daily cooling demand time-series in cooling degree-days. If `return_capacity`
        is True, returns a tuple of (cooling_demand_data, capacity_data).

    Notes
    -----
    The time shift applies uniformly across the entire spatial domain.
    More fine-grained, space- and time-dependent time zone control will
    be implemented in future versions.

    You can also specify all of the general conversion arguments
    documented in the `convert_and_aggregate` function.

    Warnings
    --------
    When using time shifts with monthly data, boundary effects may occur,
    potentially creating duplicate daily indices. Manual post-processing
    may be required to handle these edge cases.

    """
    return cutout.convert_and_aggregate(
        convert_func=convert_cooling_demand,
        threshold=threshold,
        a=a,
        constant=constant,
        hour_shift=hour_shift,
        **params,
    )


# solar thermal collectors
def convert_solar_thermal(
    ds: xr.Dataset,
    orientation: Callable,
    trigon_model: str,
    clearsky_model: str,
    c0: float,
    c1: float,
    t_store: float,
) -> xr.DataArray:
    """
    Convert solar radiation to useful thermal energy from solar collectors.

    Compute thermal energy output from solar thermal collectors considering
    collector efficiency, ambient temperature, and thermal losses.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing temperature and solar radiation data.
    orientation : Callable
        Function defining collector orientation (azimuth and tilt angles).
    trigon_model : str
        Model for computing solar irradiance on tilted surfaces.
    clearsky_model : str
        Clear-sky irradiance model for atmospheric corrections.
    c0 : float
        Optical efficiency of the solar collector (maximum efficiency).
    c1 : float
        Linear heat loss coefficient in W/(m²·K).
    t_store : float
        Storage temperature in degree Celsius for thermal loss calculations.

    Returns
    -------
    xr.DataArray
        Useful thermal energy per unit collector area with negative values
        (heat losses exceeding gains) set to zero.

    Notes
    -----
    Thermal efficiency is computed as: η = c0 - c1 * (T_store - T_ambient) / I
    where I is the incident solar irradiance. The output represents net
    useful thermal energy after accounting for optical and thermal losses.
    """
    # Convert temperature to degree Celsius.
    temperature = convert_temperature(ds)

    # Downward shortwave radiation flux is in W/m^2.
    # http://rda.ucar.edu/datasets/ds094.0/#metadata/detailed.html?_do=y
    solar_position = SolarPosition(ds)
    surface_orientation = SurfaceOrientation(ds, solar_position, orientation)
    irradiation = TiltedIrradiation(
        ds, solar_position, surface_orientation, trigon_model, clearsky_model
    )

    # Compute overall efficiency; can be negative, so need to remove negative
    # values.
    eta = c0 - c1 * (
        (t_store - temperature) / irradiation.where(irradiation != 0)
    ).fillna(0)

    # Compute output.
    output = irradiation * eta
    output = output.where(output > 0.0, 0.0)

    # Set name and units attribute.
    output = output.rename("solar thermal generation")
    output.attrs["units"] = "W/m^2"

    return output


def solar_thermal(
    cutout: Cutout,
    orientation: dict[str, float] | str | Callable = {"slope": 45.0, "azimuth": 180.0},
    trigon_model: str = "simple",
    clearsky_model: str = "simple",
    c0: float = 0.8,
    c1: float = 3.0,
    t_store: float = 80.0,
    **params: Any,
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    """
    Convert downward short-wave radiation flux and outside temperature into
    time series for solar thermal collectors.

    Mathematical model and defaults for c0, c1 based on model in [1].

    Parameters
    ----------
    cutout : cutout
    orientation : dict or str or function
        Panel orientation with slope and azimuth (units of degrees), or
        'latitude_optimal'.
    trigon_model : str
        Type of trigonometry model
    clearsky_model : str or None
        Type of clearsky model for diffuse irradiation. Either
        'simple' or 'enhanced'.
    c0, c1 : float
        Parameters for model in [1] (defaults to 0.8 and 3., respectively)
    t_store : float
        Store temperature in degree Celsius

    Note
    ----
    You can also specify all of the general conversion arguments
    documented in the `convert_and_aggregate` function.

    References
    ----------
    [1] Henning and Palzer, Renewable and Sustainable Energy Reviews 30
    (2014) 1003-1018

    """
    if not callable(orientation):
        orientation = get_orientation(orientation)

    return cutout.convert_and_aggregate(
        convert_func=convert_solar_thermal,
        orientation=orientation,
        trigon_model=trigon_model,
        clearsky_model=clearsky_model,
        c0=c0,
        c1=c1,
        t_store=t_store,
        **params,
    )


# wind
def convert_wind(
    ds: xr.Dataset,
    turbine: TurbineConfig,
    interpolation_method: Literal["logarithmic", "power"],
) -> xr.DataArray:
    """
    Convert wind speeds to wind energy generation using turbine power curve.

    Extrapolates wind speeds to turbine hub height and applies the turbine's
    power curve to compute electrical power output per unit capacity.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing wind speed data at reference heights.
    turbine : TurbineConfig
        Wind turbine configuration containing power curve data (V, POW),
        hub height, and rated power (P).
    interpolation_method : {'logarithmic', 'power'}
        Method for extrapolating wind speeds to hub height.

    Returns
    -------
    xr.DataArray
        Wind power generation per unit of installed capacity, with values
        between 0 and 1 representing capacity factors.

    Notes
    -----
    The function performs linear interpolation of the turbine power curve
    to match wind speeds at hub height with corresponding power outputs.
    """
    V, POW, hub_height, P = itemgetter("V", "POW", "hub_height", "P")(turbine)

    wnd_hub = windm.extrapolate_wind_speed(
        ds, to_height=hub_height, method=interpolation_method
    )

    def apply_power_curve(da):
        return np.interp(da, V, POW / P)

    da = xr.apply_ufunc(
        apply_power_curve,
        wnd_hub,
        input_core_dims=[[]],
        output_core_dims=[[]],
        output_dtypes=[wnd_hub.dtype],
        dask="parallelized",
    )

    # Set name and units attribute.
    da = da.rename("wind generation")
    da.attrs["units"] = "per unit of installed capacity"

    return da


def wind(
    cutout: Cutout,
    turbine: str | Path | dict,
    smooth: bool | dict = False,
    add_cutout_windspeed: bool = False,
    interpolation_method: Literal["logarithmic", "power"] = "logarithmic",
    **params: Any,
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    """
    Generate wind power generation time-series.

    Extrapolates wind speeds to turbine hub height using either logarithmic
    or power law methods, then evaluates the turbine power curve to compute
    generation per unit of installed capacity.

    Parameters
    ----------
    cutout : Cutout
        Weather data cutout containing wind speed data.
    turbine : str | Path | dict
        A turbineconfig dictionary with the keys 'hub_height' for the
        hub height and 'V', 'POW' defining the power curve.
        Alternatively a str refering to a local or remote turbine configuration
        as accepted by atlite.resource.get_windturbineconfig(). Locally stored turbine
        configurations can also be modified with this function. E.g. to setup a different hub
        height from the one used in the yaml file,one would write
                "turbine=get_windturbineconfig(“NREL_ReferenceTurbine_5MW_offshore”)|{“hub_height”:120}"
    smooth : bool | dict, default False
        If True smooth power curve with a gaussian kernel as
        determined for the Danish wind fleet to Delta_v = 1.27 and
        sigma = 2.29. A dict allows to tune these values.
    add_cutout_windspeed : bool, default False
        If True and in case the power curve does not end with a zero, will add zero power
        output at the highest wind speed in the power curve. If False, a warning will be
        raised if the power curve does not have a cut-out wind speed.
    interpolation_method : {'logarithmic', 'power'}, default 'logarithmic'
        Law to interpolate wind speed to turbine hub height. Refer to
        :py:func:`atlite.wind.extrapolate_wind_speed`.
    **params : Any
        Additional parameters for spatial/temporal aggregation. See
        `convert_and_aggregate` for available options.

    Returns
    -------
    xr.DataArray | tuple[xr.DataArray, xr.DataArray]
        Wind power generation time-series as capacity factors (per unit of
        installed capacity). If `return_capacity` is True, returns a tuple
        of (generation_data, capacity_data).

    Notes
    -----
    You can also specify all of the general conversion arguments
    documented in the `convert_and_aggregate` function.

    References
    ----------
    .. [1] Andresen G B, Søndergaard A A and Greiner M 2015 Energy 93, Part 1
       1074 – 1088. doi:10.1016/j.energy.2015.09.071

    """
    turbine = get_windturbineconfig(turbine, add_cutout_windspeed=add_cutout_windspeed)

    if smooth:
        turbine = windturbine_smooth(turbine, params=smooth)

    return cutout.convert_and_aggregate(
        convert_func=convert_wind,
        turbine=turbine,
        interpolation_method=interpolation_method,
        **params,
    )


# irradiation
def convert_irradiation(
    ds: xr.Dataset,
    orientation: Callable,
    tracking: str | None = None,
    irradiation: str = "total",
    trigon_model: str = "simple",
    clearsky_model: str = "simple",
) -> xr.DataArray:
    """
    Convert horizontal solar irradiance to tilted surface irradiance.

    Compute solar irradiance on tilted surfaces considering panel orientation,
    solar position, and atmospheric conditions. Supports fixed and tracking
    configurations.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing solar irradiance and atmospheric data.
    orientation : Callable
        Function defining surface orientation (azimuth and tilt angles).
    tracking : str or None, default None
        Solar tracking mode. If None, uses fixed orientation.
    irradiation : str, default 'total'
        Type of irradiation to compute ('total', 'direct', or 'diffuse').
    trigon_model : str, default 'simple'
        Model for computing irradiance on tilted surfaces.
    clearsky_model : str, default 'simple'
        Clear-sky irradiance model for atmospheric corrections.

    Returns
    -------
    xr.DataArray
        Solar irradiance on the tilted surface with appropriate units
        and variable name attributes.

    Notes
    -----
    The function accounts for:
    - Solar position (elevation and azimuth angles)
    - Surface orientation relative to the sun
    - Atmospheric attenuation and scattering effects
    - Tracking system behavior if specified
    """
    solar_position = SolarPosition(ds)
    surface_orientation = SurfaceOrientation(ds, solar_position, orientation, tracking)
    irradiation = TiltedIrradiation(
        ds,
        solar_position,
        surface_orientation,
        trigon_model=trigon_model,
        clearsky_model=clearsky_model,
        tracking=tracking,
        irradiation=irradiation,
    )

    # Set name and units attribute.
    irradiation = irradiation.rename(f"{irradiation} irradiation")
    irradiation.attrs["units"] = "W/m^2"

    return irradiation


def irradiation(
    cutout: Cutout,
    orientation: dict[str, float] | str | Callable,
    irradiation: str = "total",
    tracking: str | None = None,
    clearsky_model: str | None = None,
    **params: Any,
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate the total, direct, diffuse, or ground irradiation on a tilted
    surface.

    Parameters
    ----------
    cutout : Cutout
        Weather data cutout containing solar radiation data.
    orientation : str, dict, or Callable
        Panel orientation can be chosen from either
        'latitude_optimal', a constant orientation {'slope': 0.0,
        'azimuth': 0.0} or a callback function with the same signature
        as the callbacks generated by the
        'atlite.pv.orientation.make_*' functions.
    irradiation : str, default 'total'
        The irradiation quantity to be returned. Defaults to "total" for total
        combined irradiation. Other options include "direct" for direct irradiation,
        "diffuse" for diffuse irradiation, and "ground" for irradiation reflected
        by the ground via albedo. NOTE: "ground" irradiation is not calculated
        by all `trigon_model` options in the `convert_irradiation` method,
        so use with caution!
    tracking : str | None, optional
        Solar tracking configuration. Options are:
        - None : no tracking (default)
        - 'horizontal' : 1-axis horizontal tracking
        - 'tilted_horizontal' : 1-axis horizontal tracking with tilted axis
        - 'vertical' : 1-axis vertical tracking
        - 'dual' : 2-axis tracking
    clearsky_model : str | None, optional
        Either the 'simple' or the 'enhanced' Reindl clearsky
        model. The default choice of None will choose depending on
        data availability, since the 'enhanced' model also
        incorporates ambient air temperature and relative humidity.
    **params : Any
        Additional parameters for spatial/temporal aggregation. See
        `convert_and_aggregate` for available options.

    Returns
    -------
    xr.DataArray | tuple[xr.DataArray, xr.DataArray]
        The desired irradiation quantity on the tilted surface. If
        `return_capacity` is True, returns a tuple of (irradiation_data, capacity_data).

    Notes
    -----
    You can also specify all of the general conversion arguments
    documented in the `convert_and_aggregate` function.

    References
    ----------
    [1] D.T. Reindl, W.A. Beckman, and J.A. Duffie. Diffuse fraction correla-
    tions. Solar Energy, 45(1):1 – 7, 1990.

    """
    if not callable(orientation):
        orientation = get_orientation(orientation)

    return cutout.convert_and_aggregate(
        convert_func=convert_irradiation,
        orientation=orientation,
        tracking=tracking,
        irradiation=irradiation,
        clearsky_model=clearsky_model,
        **params,
    )


# solar PV
def convert_pv(
    ds: xr.Dataset,
    panel: dict,
    orientation: Callable,
    tracking: str | None,
    trigon_model: str = "simple",
    clearsky_model: str = "simple",
) -> xr.DataArray:
    """
    Convert solar irradiance to photovoltaic power generation.

    Compute PV power output considering panel characteristics, solar position,
    surface orientation, and environmental conditions including temperature
    effects on panel efficiency.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing solar irradiance and temperature data.
    panel : dict
        Solar panel configuration containing efficiency parameters and
        temperature coefficients.
    orientation : Callable
        Function defining panel orientation (azimuth and tilt angles).
    tracking : str or None
        Solar tracking mode. If None, uses fixed orientation.
    trigon_model : str, default 'simple'
        Model for computing irradiance on tilted surfaces.
    clearsky_model : str, default 'simple'
        Clear-sky irradiance model for atmospheric corrections.

    Returns
    -------
    xr.DataArray
        PV power generation per unit of installed capacity with values
        between 0 and 1 representing capacity factors.

    Notes
    -----
    The function accounts for:
    - Solar irradiance on tilted panel surface
    - Temperature-dependent panel efficiency
    - Panel characteristics and specifications
    - Tracking system behavior if specified
    """
    solar_position = SolarPosition(ds)
    surface_orientation = SurfaceOrientation(ds, solar_position, orientation, tracking)
    irradiation = TiltedIrradiation(
        ds,
        solar_position,
        surface_orientation,
        trigon_model=trigon_model,
        clearsky_model=clearsky_model,
        tracking=tracking,
    )
    solar_panel = SolarPanelModel(ds, irradiation, panel)

    # Set name and units attribute.
    solar_panel = solar_panel.rename("solar PV generation")
    solar_panel.attrs["units"] = "per unit of installed capacity"

    return solar_panel


def pv(
    cutout: Cutout,
    panel: str | Path | dict,
    orientation: dict[str, float] | str | Callable,
    tracking: str | None = None,
    clearsky_model: str | None = None,
    **params: Any,
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    """
    Generate photovoltaic (PV) power generation time-series.

    Converts solar irradiance and ambient temperature into PV generation
    using panel electrical models and orientation specifications.

    Parameters
    ----------
    cutout : Cutout
        The weather data cutout containing solar radiation and temperature data.
    panel : str | Path | dict
        Panel configuration dictionary with electrical model parameters,
        or a string/Path reference to a stored panel configuration file.
    orientation : dict[str, float] | str | Callable
        Panel orientation specification. Can be 'latitude_optimal', a constant
        orientation dict with 'slope' and 'azimuth' keys (in degrees), or a
        callable function for dynamic orientation.
    tracking : str | None, optional
        Solar tracking configuration. Options include:
        - None: No tracking (default)
        - 'horizontal': 1-axis horizontal tracking
        - 'tilted_horizontal': 1-axis tilted horizontal tracking
        - 'vertical': 1-axis vertical tracking
        - 'dual': 2-axis tracking
    clearsky_model : str | None, optional
        Clearsky model for diffuse irradiation calculation. Either 'simple'
        or 'enhanced' Reindl model. If None, automatically selects based on
        data availability ('enhanced' requires temperature and humidity).
    **params : Any
        Additional parameters for spatial/temporal aggregation. See
        `convert_and_aggregate` for available options.

    Returns
    -------
    xr.DataArray | tuple[xr.DataArray, xr.DataArray]
        PV power generation time-series as capacity factors (per unit of
        installed capacity). If `return_capacity` is True, returns a tuple
        of (generation_data, capacity_data).

    Notes
    -----
    You can specify all general conversion arguments documented in the
    `convert_and_aggregate` function for spatial and temporal aggregation.

    References
    ----------
    [1] Soteris A. Kalogirou. Solar Energy Engineering: Processes and Systems,
    pages 49–117,469–516. Academic Press, 2009. ISBN 0123745012.

    [2] D.T. Reindl, W.A. Beckman, and J.A. Duffie. Diffuse fraction correla-
    tions. Solar Energy, 45(1):1 – 7, 1990.

    [3] Hans Georg Beyer, Gerd Heilscher and Stefan Bofinger. A Robust Model
    for the MPP Performance of Different Types of PV-Modules Applied for
    the Performance Check of Grid Connected Systems, Freiburg, June 2004.
    Eurosun (ISES Europe Solar Congress).

    """
    if isinstance(panel, (str | Path)):
        panel = get_solarpanelconfig(panel)
    if not callable(orientation):
        orientation = get_orientation(orientation)

    return cutout.convert_and_aggregate(
        convert_func=convert_pv,
        panel=panel,
        orientation=orientation,
        tracking=tracking,
        clearsky_model=clearsky_model,
        **params,
    )


# solar CSP
def convert_csp(ds: xr.Dataset, installation: dict) -> xr.DataArray:
    """
    Convert solar radiation data to concentrated solar power (CSP) generation.

    This function converts direct solar radiation data into CSP generation
    considering the efficiency characteristics of the specific installation
    and solar position.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing solar radiation data.
    installation : dict
        CSP installation configuration containing efficiency data and
        technology specifications.

    Returns
    -------
    xr.DataArray
        CSP generation per unit of installed capacity with time and spatial
        dimensions preserved.

    Notes
    -----
    The function handles different CSP technologies:
    - 'parabolic trough': Uses direct horizontal irradiance (DHI)
    - 'solar tower': Uses direct normal irradiance (DNI)

    Output is normalized by reference irradiance and clipped to a maximum
    of 1.0 per unit capacity.
    """
    solar_position = SolarPosition(ds)

    tech = installation["technology"]
    match tech:
        case "parabolic trough":
            irradiation = ds["influx_direct"]
        case "solar tower":
            irradiation = cspm.calculate_dni(ds, solar_position)
        case _:
            raise ValueError(f'Unknown CSP technology option "{tech}".')

    # Determine solar_position dependent efficiency for each grid cell and time step.
    efficiency = installation["efficiency"].interp(
        altitude=solar_position["altitude"], azimuth=solar_position["azimuth"]
    )

    # Thermal system output.
    da = efficiency * irradiation

    # Output relative to reference irradiance.
    da /= installation["r_irradiance"]

    # Limit output to max of reference irradiance.
    da = da.clip(max=1.0)

    # Fill NaNs originating from DNI or solar positions outside efficiency bounds.
    da = da.fillna(0.0)

    # Set name and units attribute.
    da = da.rename("csp generation")
    da.attrs["units"] = "per unit of installed capacity"

    return da


def csp(
    cutout: Cutout,
    installation: str | Path | dict,
    technology: str | None = None,
    **params: Any,
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    """
    Convert downward shortwave direct radiation into CSP generation time-series.

    Parameters
    ----------
    cutout : Cutout
        Weather data cutout containing solar radiation data.
    installation : str | Path | dict
        CSP installation details determining the solar field efficiency dependent on
        the local solar position. Can be either the name of one of the standard
        installations provided through `atlite.cspinstallationsPanel` or an
        xarray.DataArray with 'azimuth' (in rad) and 'altitude' (in rad) coordinates
        and an 'efficiency' (in p.u.) entry.
    technology : str | None, optional
        Overwrite CSP technology from the installation configuration. The technology
        affects which direct radiation is considered. Either 'parabolic trough' (DHI)
        or 'solar tower' (DNI).
    **params : Any
        Additional parameters for spatial/temporal aggregation. See
        `convert_and_aggregate` for available options.

    Returns
    -------
    xr.DataArray | tuple[xr.DataArray, xr.DataArray]
        CSP generation time-series or capacity factors. If `return_capacity`
        is True, returns a tuple of (generation_data, capacity_data).

    Notes
    -----
    You can also specify all of the general conversion arguments
    documented in the `convert_and_aggregate` function.

    References
    ----------
    [1] Tobias Hirsch (ed.). SolarPACES Guideline for Bankable STE Yield Assessment,
    IEA Technology Collaboration Programme SolarPACES, 2017.
    URL: https://www.solarpaces.org/csp-research-tasks/task-annexes-iea/task-i-solar-thermal-electric-systems/solarpaces-guideline-for-bankable-ste-yield-assessment/

    [2] Tobias Hirsch (ed.). CSPBankability Project Report, DLR, 2017.
    URL: https://www.dlr.de/sf/en/desktopdefault.aspx/tabid-11126/19467_read-48251/

    """
    if isinstance(installation, (str | Path)):
        installation = get_cspinstallationconfig(installation)

    # Overwrite technology if specified.
    if technology is not None:
        installation["technology"] = technology

    return cutout.convert_and_aggregate(
        convert_func=convert_csp,
        installation=installation,
        **params,
    )


# hydro
def convert_runoff(ds: xr.Dataset, weight_with_height: bool = True) -> xr.DataArray:
    """
    Convert runoff data with optional height weighting.

    Extract runoff data from the dataset and optionally weight it by
    topographic height to account for elevation effects on runoff generation.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing runoff and height data.
    weight_with_height : bool, default True
        Whether to weight runoff by topographic height.

    Returns
    -------
    xr.DataArray
        Runoff data, optionally weighted by height.

    Notes
    -----
    Height weighting is commonly used in hydrological modeling to account
    for orographic effects on precipitation and subsequent runoff generation.
    """
    runoff = ds["runoff"]

    if weight_with_height:
        runoff = runoff * ds["height"]

    return runoff


def runoff(
    cutout: Cutout,
    smooth: int | bool | None = None,
    lower_threshold_quantile: float | bool | None = None,
    normalize_using_yearly: pd.Series | None = None,
    **params: Any,
) -> xr.DataArray | tuple[xr.DataArray, xr.DataArray]:
    """
    Convert runoff data with optional smoothing and normalization.

    This function processes runoff data from weather reanalysis datasets
    with various post-processing options including temporal smoothing,
    threshold filtering, and normalization to observed data.

    Parameters
    ----------
    cutout : Cutout
        Cutout object containing runoff data and spatial metadata.
    smooth : int, bool, or None, default None
        Temporal smoothing window in hours. If True, defaults to 168 hours
        (one week). If None, no smoothing is applied.
    lower_threshold_quantile : float, bool, or None, default None
        Quantile threshold below which runoff values are set to zero.
        If True, defaults to 0.005 (0.5th percentile). If None, no
        threshold filtering is applied.
    normalize_using_yearly : pd.Series or None, default None
        Annual runoff data for normalization. Index should contain years
        and values the corresponding annual runoff volumes.
    **params : Any
        Additional parameters passed to the `convert_and_aggregate` method.

    Returns
    -------
    xr.DataArray or tuple[xr.DataArray, xr.DataArray]
        Processed runoff time series. Returns tuple if `return_capacity`
        is specified in params.

    Notes
    -----
    The function applies processing steps in sequence:
    1. Convert raw runoff data using `convert_runoff`
    2. Apply temporal smoothing if requested
    3. Filter values below threshold quantile if specified
    4. Normalize to yearly observations if provided

    Normalization requires at least one full year of data and uses only
    overlapping years between the dataset and reference data.
    """
    result = cutout.convert_and_aggregate(convert_func=convert_runoff, **params)

    if smooth is not None:
        if smooth is True:
            smooth = 24 * 7
        if "return_capacity" in params:
            result = result[0].rolling(time=smooth, min_periods=1).mean(), result[1]
        else:
            result = result.rolling(time=smooth, min_periods=1).mean()

    if lower_threshold_quantile is not None:
        if lower_threshold_quantile is True:
            lower_threshold_quantile = 5e-3
        lower_threshold = pd.Series(result.values.ravel()).quantile(
            lower_threshold_quantile
        )
        result = result.where(result >= lower_threshold, 0.0)

    if normalize_using_yearly is not None:
        normalize_using_yearly_i = normalize_using_yearly.index
        if isinstance(normalize_using_yearly_i, pd.DatetimeIndex):
            normalize_using_yearly_i = normalize_using_yearly_i.year
        else:
            normalize_using_yearly_i = normalize_using_yearly_i.astype(int)

        years = (
            pd.Series(pd.to_datetime(result.coords["time"].values).year)
            .value_counts()
            .loc[lambda x: x > 8700]
            .index.intersection(normalize_using_yearly_i)
        )
        assert len(years), "Need at least a full year of data (more is better)"
        years_overlap = slice(str(min(years)), str(max(years)))

        dim = result.dims[1 - result.get_axis_num("time")]
        result *= (
            xr.DataArray(normalize_using_yearly.loc[years_overlap].sum(), dims=[dim])
            / result.sel(time=years_overlap).sum("time")
        ).reindex(countries=result.coords["countries"])

    return result


def hydro(
    cutout: Cutout,
    plants: pd.DataFrame,
    hydrobasins: str | gpd.GeoDataFrame,
    flowspeed: float = 1,
    weight_with_height: bool = False,
    show_progress: bool = False,
    **kwargs: Any,
) -> xr.DataArray:
    """
    Generate hydropower inflow time-series for plants using catchment basins.

    Computes inflow time-series by aggregating surface runoff over catchment
    basins and routing water flows to hydropower plant locations.

    Parameters
    ----------
    cutout : Cutout
        The weather data cutout containing surface runoff data.
    plants : pd.DataFrame
        Hydropower plants (run-of-river or dams) with 'lon' and 'lat' columns
        specifying plant locations.
    hydrobasins : str | gpd.GeoDataFrame
        HydroBASINS dataset for catchment delineation. Can be a filename
        or GeoDataFrame containing basin geometries.
    flowspeed : float, default 1
        Average water flow speed in m/s for estimating travel time from
        basin to plant location.
    weight_with_height : bool, default False
        Whether to weight surface runoff by elevation (recommended for
        coarser spatial resolution).
    show_progress : bool, default False
        Whether to display progress bars during computation.
    **kwargs : Any
        Additional keyword arguments passed to runoff computation.

    Returns
    -------
    xr.DataArray
        Hydropower inflow time-series for each plant, accounting for
        catchment aggregation and flow routing delays.

    References
    ----------
    [1] Liu, Hailiang, et al. "A validated high-resolution hydro power
    time-series model for energy systems analysis." arXiv preprint
    arXiv:1901.08476 (2019).

    [2] Lehner, B., Grill G. (2013): Global river hydrography and network
    routing: baseline data and new approaches to study the world’s large river
    systems. Hydrological Processes, 27(15): 2171–2186. Data is available at
    www.hydrosheds.org.

    """
    basins = hydrom.determine_basins(plants, hydrobasins, show_progress=show_progress)

    matrix = cutout.indicatormatrix(basins.shapes)
    # Compute the average surface runoff in each basin.
    # Fix NaN and Inf values to 0.0 to avoid numerical issues.
    matrix_normalized = np.nan_to_num(
        matrix / matrix.sum(axis=1), nan=0.0, posinf=0.0, neginf=0.0
    )
    runoff = cutout.runoff(
        matrix=matrix_normalized,
        index=basins.shapes.index,
        weight_with_height=weight_with_height,
        show_progress=show_progress,
        **kwargs,
    )
    # The hydrological parameters are in units of "m of water per day" and so
    # they should be multiplied by 1000 and the basin area to convert to m3
    # d-1 = m3 h-1 / 24.
    runoff *= xr.DataArray(basins.shapes.to_crs(dict(proj="cea")).area)

    return hydrom.shift_and_aggregate_runoff_for_plants(
        basins, runoff, flowspeed, show_progress
    )


def convert_line_rating(
    ds: xr.Dataset,
    psi: float,
    R: float,
    D: float = 0.028,
    Ts: float = 373,
    epsilon: float = 0.6,
    alpha: float = 0.6,
) -> xr.DataArray:
    """
    Convert weather data to dynamic line rating time series.

    Calculate maximum allowable current for overhead transmission lines
    based on thermal balance considering wind cooling, solar heating,
    and conductor characteristics following IEEE-738 standard.

    Parameters
    ----------
    ds : xr.Dataset
        Weather dataset containing temperature, wind speed, wind direction,
        solar radiation, and elevation data for grid cells overlapping
        with the transmission line.
    psi : float
        Azimuth angle of the transmission line in degrees, measured as
        the angle from north (0° = north, 90° = east, 180° = south, 270° = west).
    R : float
        Electrical resistance of the conductor in Ω/m at maximum allowed
        temperature Ts.
    D : float, default 0.028
        Conductor diameter in meters.
    Ts : float, default 373
        Maximum allowed surface temperature in Kelvin (typically 100°C = 373 K).
    epsilon : float, default 0.6
        Conductor emissivity for radiative heat loss (dimensionless, 0-1).
    alpha : float, default 0.6
        Conductor absorptivity for solar radiation (dimensionless, 0-1).

    Returns
    -------
    xr.DataArray
        Maximum allowable current capacity per timestep in Amperes, with
        time and spatial dimensions preserved from input dataset.

    Notes
    -----
    The calculation is based on IEEE Std 738™-2012 with the following
    simplifications and assumptions:

    1. Wind speeds are taken at 100 meters height, though transmission
       lines are typically at 50-60 meters height.
    2. Solar heat influx is proportional to shortwave radiation flux.
    3. Solar incidence angle is assumed to be 90 degrees.

    References
    ----------
    [1] IEEE Std 738™-2012 (Revision of IEEE Std 738-2006/Incorporates IEEE Std
        738-2012/Cor 1-2013), IEEE Standard for Calculating the Current-Temperature
        Relationship of Bare Overhead Conductors, p. 72.

    """
    Ta = ds["temperature"]
    Tfilm = (Ta + Ts) / 2
    T0 = 273.15

    # 1. Convective loss, at first forced convection.
    V = ds["wnd100m"]  # Typically ironmen are about 40-60 meters high.
    mu = (1.458e-6 * Tfilm**1.5) / (
        Tfilm + 383.4 - T0
    )  # Dynamic viscosity of air (13a).
    H = ds["height"]
    rho = (1.293 - 1.525e-4 * H + 6.379e-9 * H**2) / (
        1 + 0.00367 * (Tfilm - T0)
    )  # (14a).

    reynold = D * V * rho / mu

    k = (
        2.424e-2 + 7.477e-5 * (Tfilm - T0) - 4.407e-9 * (Tfilm - T0) ** 2
    )  # Thermal conductivity.
    anglediff = ds["wnd_azimuth"] - radians(psi)
    Phi = absolute(mod(anglediff + pi / 2, pi) - pi / 2)
    K = (
        1.194 - cos(Phi) + 0.194 * cos(2 * Phi) + 0.368 * sin(2 * Phi)
    )  # Wind direction factor.

    Tdiff = Ts - Ta
    qcf1 = K * (1.01 + 1.347 * reynold**0.52) * k * Tdiff  # (3a) in [1].
    qcf2 = K * 0.754 * reynold**0.6 * k * Tdiff  # (3b) in [1].

    qcf = maximum(qcf1, qcf2)

    # Natural convection.
    qcn = 3.645 * sqrt(rho) * D**0.75 * Tdiff**1.25

    # Convection loss is the max between forced and natural.
    qc = maximum(qcf, qcn)

    # 2. Radiated loss.
    qr = 17.8 * D * epsilon * ((Ts / 100) ** 4 - (Ta / 100) ** 4)

    # 3. Solar radiance heat gain.
    Q = ds["influx_direct"]  # Assumption: this is short wave and not heat influx.
    A = D * 1  # Projected area of conductor in square meters.

    if isinstance(ds, dict):
        Position = namedtuple("solarposition", ["altitude", "azimuth"])
        solar_position = Position(ds["solar_altitude"], ds["solar_azimuth"])
    else:
        solar_position = SolarPosition(ds)
    Phi_s = arccos(
        cos(solar_position.altitude) * cos((solar_position.azimuth) - radians(psi))
    )

    qs = alpha * Q * A * sin(Phi_s)

    Imax = sqrt((qc + qr - qs) / R)
    Imax = Imax.min("spatial") if isinstance(Imax, xr.DataArray) else Imax

    # Set name and units attribute.
    Imax = Imax.rename("maximum line current")
    Imax.attrs["units"] = "A"

    return Imax


def line_rating(
    cutout: Cutout,
    shapes: gpd.GeoSeries,
    line_resistance: float | pd.Series,
    show_progress: bool = False,
    dask_kwargs: dict[str, Any] | None = None,
    **params: Any,
) -> xr.DataArray:
    """
    Create dynamic line rating time series based on IEEE-738 standard.

    Calculate maximum allowable current for overhead transmission lines
    using thermal balance equations. The steady-state capacity is derived
    from the balance between heat losses (radiation and convection) and
    heat gains (solar influx and conductor resistance). For more information
    on assumptions and modifications see ``convert_line_rating``.

    Parameters
    ----------
    cutout : Cutout
        The weather data cutout containing temperature, wind, and radiation data.
    shapes : gpd.GeoSeries
        Geographic line shapes representing transmission lines.
    line_resistance : float | pd.Series
        Electrical resistance of transmission lines in Ohm/meter. Can be a
        single value or series with per-line values.
    show_progress : bool, default False
        Whether to display a progress bar during computation.
    dask_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments passed to `dask.compute()`.
    **params : Any
        Additional conductor parameters for thermal calculations. Default values:
        - D : 0.028 (conductor diameter in meters)
        - Ts : 373 (maximum allowed surface temperature in Kelvin)
        - epsilon : 0.6 (conductor emissivity)
        - alpha : 0.6 (conductor absorptivity)

    Returns
    -------
    xr.DataArray
        Dynamic thermal capacity time-series with dimensions (time x lines)
        giving maximum allowable current in Amperes for each transmission line.

    Examples
    --------
    >>> import pypsa
    >>> import xarray as xr
    >>> import atlite
    >>> import numpy as np
    >>> import geopandas as gpd
    >>> from shapely.geometry import Point, LineString as Line

    >>> n = pypsa.examples.scigrid_de()
    >>> n.calculate_dependent_values()
    >>> x = n.buses.x
    >>> y = n.buses.y
    >>> buses = n.lines[["bus0", "bus1"]].values
    >>> shapes = [Line([Point(x[b0], y[b0]), Point(x[b1], y[b1])]) for (b0, b1) in buses]
    >>> shapes = gpd.GeoSeries(shapes, index=n.lines.index)

    >>> cutout = atlite.Cutout('test', x=slice(x.min(), x.max()), y=slice(y.min(), y.max()),
                            time='2020-01-01', module='era5', dx=1, dy=1)
    >>> cutout.prepare()

    >>> i = cutout.line_rating(shapes, n.lines.r/n.lines.length)
    >>> v = xr.DataArray(n.lines.v_nom, dims='name')
    >>> s = np.sqrt(3) * i * v / 1e3  # In MW

    References
    ----------
    [1] IEEE Std 738™-2012 (Revision of IEEE Std 738-2006/Incorporates IEEE Std
        738-2012/Cor 1-2013), IEEE Standard for Calculating the Current-Temperature
        Relationship of Bare Overhead Conductors, p. 72.

    """
    # Handle mutable default arguments.
    if dask_kwargs is None:
        dask_kwargs = {}

    if not isinstance(shapes, gpd.GeoSeries):
        shapes = gpd.GeoSeries(shapes).rename_axis("dim_0")

    intersection_matrix = cutout.intersectionmatrix(shapes)
    rows, cols = intersection_matrix.nonzero()

    data = cutout.data.stack(spatial=["y", "x"])

    def get_azimuth(shape):
        coords = np.array(shape.coords)
        start = coords[0]
        end = coords[-1]
        return np.arctan2(start[0] - end[0], start[1] - end[1])

    azimuth = shapes.apply(get_azimuth)
    azimuth = azimuth.where(azimuth >= 0, azimuth + np.pi)

    params.setdefault("D", 0.028)
    params.setdefault("Ts", 373)
    params.setdefault("epsilon", 0.6)
    params.setdefault("alpha", 0.6)

    df = pd.DataFrame({"psi": azimuth, "R": line_resistance}).assign(**params)

    assert df.notnull().all().all(), "NaN values encountered."
    assert df.columns.equals(pd.Index(["psi", "R", "D", "Ts", "epsilon", "alpha"]))

    dummy = xr.DataArray(np.full(len(data.time), np.nan), coords=(data.time,))
    res = []
    for i, row in enumerate(df.itertuples(index=False)):
        cells_i = cols[rows == i]
        if cells_i.size:
            ds = data.isel(spatial=cells_i)
            res.append(delayed(convert_line_rating)(ds, *row))
        else:
            res.append(dummy)
    if show_progress:
        with ProgressBar(minimum=2):
            res = compute(res, **dask_kwargs)
    else:
        res = compute(res, **dask_kwargs)

    return xr.concat(*res, dim=df.index).assign_attrs(units="A")
