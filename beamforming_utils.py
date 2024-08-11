import torch
import numpy as np
import jax.numpy as jnp


def get_receive_beamforming(domain, time_axis, positions, output_data, signal, carrier_signal, signal_delay, c0=1500):
    """
    Get the receive beamforming data.

    Parameters
    ----------
    domain : jwave.geometry.Domain
        Spatial domain.
    time_axis : jwave.geometry.TimeAxis
        Time axis.
    positions : np.ndarray
        Transducer element positions.
    output_data : np.ndarray
        Pressure data at the receiver positions.
    signal : np.ndarray
        Emitted source signal at transmitter positions.
    carrier_signal : np.ndarray
        Carrier source signal.
    signal_delay : int
        Signal delay in timepoints.
    c0 : float
        Reference sound speed in meters/second.

    Returns
    -------
    receive : np.ndarray
        Receive beamforming data.
    """

    slope = signal_delay * time_axis.dt * c0 / domain.dx[0].item()
    oblique_factor = (slope*slope+1)**0.5
    s_stack_t = torch.from_numpy(np.asarray(jnp.vstack([signal[i] for i in range(positions.shape[1])]))).to("cuda:0")
    s_stack_t /= torch.max(s_stack_t).item()

    def compute_time_delays_for_point(x1: np.ndarray, x: int, delta_y: int, c: float = c0):
        scaled_y = delta_y * domain.dx[1]
        scaled_dx = (x1 - x) * domain.dx[0]
        return np.round((scaled_y * oblique_factor + np.sqrt(scaled_y*scaled_y + scaled_dx * scaled_dx)) / (c  * time_axis.dt)).astype(int).tolist()

    s_t = torch.from_numpy(np.asarray(carrier_signal)).to("cuda:0")
    s_t /= torch.max(s_t).item()
    output_data_t = torch.from_numpy(np.asarray(output_data)).to("cuda:0")

    nelements = positions.shape[1]
    transducer_x_start = positions[0][0]
    transducer_y = positions[1][0]
    def compute_signal(pt_x, pt_y):
        delta_y = abs(transducer_y - pt_y)
        delays = compute_time_delays_for_point(positions[0], pt_x, delta_y)
        signal = torch.zeros_like(output_data_t[:, 0])
        slanted_x_coord = int(pt_x - slope * delta_y)
        if slanted_x_coord > transducer_x_start + nelements - 1 or slanted_x_coord < transducer_x_start:
            return 0.0
        for i in range(len(delays)):
            if abs(positions[0][i] - (pt_x - slope * delta_y)) < delta_y:
                delta = delays[i]
                signal[:-delta] += output_data_t[delta:, i]
        return (torch.dot(signal, s_stack_t[slanted_x_coord - transducer_x_start]) * time_axis.dt).item()

    receive = np.array([[compute_signal(i, j) for j in range(0, domain.N[1])] for i in range(0, domain.N[0])])
    return receive