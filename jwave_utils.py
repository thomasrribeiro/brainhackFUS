import numpy as np
import jax.numpy as jnp
from jax import jit
from jwave import FourierSeries
from jwave.geometry import Domain, Medium, Sources
from jwave.acoustics import simulate_wave_propagation
from jwave.signal_processing import gaussian_window


def get_domain(N, dx):
    """
    Get the spatial domain.

    Parameters
    ----------
    N : np.ndarray
        Number of grid points in each dimension.
    dx : np.ndarray
        Grid spacing in each dimension in meters.

    Returns
    -------
    domain : jwave.geometry.Domain
        Spatial domain.
    """
    domain = Domain((N[0],N[1]), (dx[0], dx[1]))
    return domain


def get_point_medium(domain, c0=1500, rho0=1000, pml_size=20):
    """
    Get an acoustic medium with a single point scatterer.

    Parameters
    ----------
    domain : jwave.geometry.Domain
        Spatial domain.
    c0 : float
        Reference speed of sound in m/s.
    rho0 : float
        Reference density in kg/m^3.
    pml_size : int
        Size of the PML in grid points.

    Returns
    -------
    medium : jwave.medium.Medium
        Medium.
    """

    np.random.seed(28)
    N = domain.N

    # define a random distribution of scatterers for the medium
    background_map_mean = 1
    background_map_std = 0.008
    background_map = background_map_mean + background_map_std * np.random.randn(N[0], N[1])
    sound_speed = c0 * np.ones(N) * background_map
    density = rho0 * np.ones(N) * background_map

    # define highly scattering region
    scatterer_radius = 2 # radius of scatterers [grid points]
    scatterer_contrast = 1.1 # contrast of scatterers
    scatterer_positions = np.array([[N[0]//2, N[1]//2]], dtype=int)
    scatterer_map = np.zeros(N)
    x, y = np.ogrid[:N[0], :N[1]]
    for scatterer_position in scatterer_positions:
        scatterer_map[(x - scatterer_position[0])**2 + (y - scatterer_position[1])**2 <= (scatterer_radius)**2] = 1
    sound_speed[scatterer_map == 1] = c0*scatterer_contrast
    density[scatterer_map == 1] = rho0*scatterer_contrast

    # define medium
    sound_speed = FourierSeries(np.expand_dims(sound_speed, -1), domain)
    density = FourierSeries(np.expand_dims(density, -1), domain)
    medium = Medium(domain=domain, sound_speed=sound_speed, density=density, pml_size=pml_size)

    return medium


def get_plane_wave_excitation(domain, time_axis, magnitude, frequency, positions):
    """
    Get a plane wave excitation.
    
    Parameters
    ----------
    domain : jwave.geometry.Domain
        Spatial domain.
    time_axis : jwave.geometry.TimeAxis
        Time axis.
    magnitude : float
        Magnitude of the excitation.
    frequency : float
        Frequency of the excitation.
    positions : np.ndarray
        Positions of the sources in grid points.

    Returns
    -------
    sources : jwave.geometry.Sources
        Sources.
    """
    
    t = jnp.arange(0, time_axis.t_end, time_axis.dt)
    s = magnitude * jnp.sin(2 * jnp.pi * frequency * t)
    variance = 2/frequency
    mean = 3*variance
    s = gaussian_window(s, t, mean, variance)
    sources = Sources(
        positions=tuple(map(tuple, positions)),
        signals=jnp.vstack([s for _ in range(positions.shape[1])]),
        dt=time_axis.dt,
        domain=domain,
    )

    return sources


def get_data(medium, time_axis, sources, sensor_positions):
    """
    Get the ultrasound pressure data.

    Parameters
    ----------
    medium : jwave.medium.Medium
        Medium.
    time_axis : jwave.geometry.TimeAxis
        Time axis.
    sources : jwave.geometry.Sources
        Sources.
    sensor_positions : np.ndarray
        Positions of the sensors in grid points.

    Returns
    -------
    pressure : jwave.geometry.Pressure
        Pressure.
    data : np.ndarray
        Pressure data.
    """

    # run simulation
    @jit
    def compiled_simulator(sources):
        pressure = simulate_wave_propagation(medium, time_axis, sources=sources)
        return pressure
    pressure = compiled_simulator(sources)

    # get pressure data
    data = np.squeeze(pressure.params[:, sensor_positions[0], sensor_positions[1]])
    
    return pressure, data