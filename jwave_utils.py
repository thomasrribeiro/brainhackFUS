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


def get_background_map(domain, c0=1500, rho0=1000, seed=28):

    np.random.seed(seed)
    N = domain.N

    # define a random distribution of scatterers for the medium
    background_map_mean = 1
    background_map_std = 0.004
    background_map = background_map_mean + background_map_std * np.random.randn(N[0], N[1])
    sound_speed = c0 * np.ones(N) * background_map
    density = rho0 * np.ones(N) * background_map

    return sound_speed, density

def get_single_scatterer(domain, sound_speed, density, c0, rho0, 
                         scatterer_radius, scatterer_contrast):
    """
    Get a single scatterer.

    Parameters
    ----------
    domain : jwave.geometry.Domain
        Spatial domain.
    sound_speed : np.ndarray
        Speed of sound.
    density : np.ndarray
        Density.
    scatterer_radius : int
        Radius of scatterers in grid points.
    scatterer_contrast : float
        Contrast of scatterers.
    c0 : float
        Reference speed of sound in m/s.
    rho0 : float
        Reference density in kg/m^3.

    Returns
    -------
    sound_speed : np.ndarray
        Speed of sound.
    density : np.ndarray
        Density.
    """

    N = domain.N

    # define highly scattering region
    scatterer_positions = np.array([[N[0]//2, N[1]//2]], dtype=int)
    scatterer_map = np.zeros(N)
    x, y = np.ogrid[:N[0], :N[1]]
    for scatterer_position in scatterer_positions:
        scatterer_map[(x - scatterer_position[0])**2 + (y - scatterer_position[1])**2 <= (scatterer_radius)**2] = 1
    sound_speed[scatterer_map == 1] = c0*scatterer_contrast
    density[scatterer_map == 1] = rho0*scatterer_contrast

    return sound_speed, density

def get_homogeneous_medium(domain, c0=1500, rho0=1000, pml_size=20):
    """
    Get a homogeneous acoustic medium.

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
    sound_speed = c0 * np.ones(domain.N)
    density = rho0 * np.ones(domain.N)
    sound_speed = FourierSeries(np.expand_dims(sound_speed, -1), domain)
    density = FourierSeries(np.expand_dims(density, -1), domain)
    medium = Medium(domain=domain, sound_speed=sound_speed, density=density, pml_size=pml_size)
    return medium

def get_point_medium(domain, c0=1500, rho0=1000, pml_size=20, 
                     scatterer_radius=2, scatterer_contrast=1.1):
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
    sound_speed, density = get_background_map(domain, c0, rho0)
    sound_speed, density = get_single_scatterer(domain, sound_speed, density, c0, rho0, scatterer_radius, scatterer_contrast)
    sound_speed = FourierSeries(np.expand_dims(sound_speed, -1), domain)
    density = FourierSeries(np.expand_dims(density, -1), domain)
    medium = Medium(domain=domain, sound_speed=sound_speed, density=density, pml_size=pml_size)
    return medium

def get_skull_point_medium(domain, skull_slice, c0=1500, rho0=1000, pml_size=20,
                           scatterer_radius=2, scatterer_contrast=1.1):
    """
    Get an acoustic medium with a skull and single point scatterer.

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

    N = domain.N
    sound_speed, density = get_background_map(domain, c0, rho0)
    sound_speed, density = get_single_scatterer(domain, sound_speed, density, c0, rho0, scatterer_radius, scatterer_contrast)

    skull_mask = skull_slice > 20000
    sound_speed[skull_mask] = 2700
    density[skull_mask] = 1800

    sound_speed = FourierSeries(np.expand_dims(sound_speed, -1), domain)
    density = FourierSeries(np.expand_dims(density, -1), domain)
    medium = Medium(domain=domain, sound_speed=sound_speed, density=density, pml_size=pml_size)
    return medium


def get_plane_wave_excitation(domain, time_axis, magnitude, frequency, positions, delay_s=0):
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
    delay_s : float
        Delay in seconds.

    Returns
    -------
    sources : jwave.geometry.Sources
        Sources.
    """
    
    t = jnp.arange(0, time_axis.t_end, time_axis.dt)
    s = magnitude * jnp.sin(2 * jnp.pi * frequency * t)
    variance = 1/frequency
    mean = 3*variance
    ss = []
    for i in range(positions.shape[1]):
        if delay_s < 0:
            ss.append(gaussian_window(s, t, mean + (i-64) * delay_s * time_axis.dt, variance))
        elif delay_s > 0:
            ss.append(gaussian_window(s, t, mean + i * delay_s * time_axis.dt, variance))
        else:
            ss.append(gaussian_window(s, t, mean, variance))
    
    sources = Sources(
        positions=tuple(map(tuple, positions)),
        signals=jnp.vstack([ss[i] for i in range(positions.shape[1])]),
        dt=time_axis.dt,
        domain=domain,
    )

    return sources, ss, s


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

def compute_time_delays_for_point(x1: np.ndarray, x: int, delta_y: int, c: float, dx: np.ndarray) -> np.ndarray:
    """
    Compute the time delays from a point x to every sensor.

    Parameters
    ----------
    x1 : np.ndarray
        x-coordinates of the sensors in grid points.
    x : int
        Position of the point of interest in grid points.
    delta_y : int
        Difference in y-coordinates of the sensors and the point of interest, in grid points.
    c : float
        Speed of sound in m/s.
    dx : np.ndarray
        Grid spacing in each dimension in meters.

    Returns
    -------
    time_delays : np.ndarray
        Time delays, in seconds.
    """
    scaled_y = delta_y * dx[1]
    scaled_dx = (x1 - x) * dx[0]
    return (scaled_y + np.sqrt(scaled_y*scaled_y + scaled_dx * scaled_dx)) / c

def compute_signal(data, element_positions, pt_x, delta_y, time, time_axis, c, dx):
    """
    Compute the signal at a given point.

    Parameters
    ----------
    data : np.ndarray
        Pressure data, shape (num_timesteps, num_sensors)
    element_positions : np.ndarray
        Positions of the sensors in grid points.
    pt_x : int
        Position of the point of interest in grid points.
    delta_y : int
        Difference in y-coordinates of the sensors and the point of interest, in grid points.
    time : int
        Time, in time steps.
    time_axis : jwave.geometry.TimeAxis
        Time axis.
    c : float
        Speed of sound in m/s.
    dx : np.ndarray
        Grid spacing in each dimension in meters.

    Returns
    -------
    signal : float
        Signal at the point of interest.
    """
    times = time - np.round(compute_time_delays_for_point(element_positions[0], pt_x, delta_y, c, dx) / time_axis.dt).astype(int)
    augmented_times = np.stack([times, np.arange(0, times.shape[0])])
    augmented_times = augmented_times[:, 0 < augmented_times[0]]
    augmented_times = augmented_times[:, augmented_times[0] < data.shape[0]].tolist()
    return np.sum(data[augmented_times[0], augmented_times[1]]).item() * time_axis.dt