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
    s_stack_t = np.asarray(jnp.vstack([signal[i] for i in range(positions.shape[1])]))
    s_stack_t = s_stack_t / np.max(s_stack_t)

    def compute_time_delays_for_point(x1: np.ndarray, x: int, delta_y: int, c: float = c0):
        scaled_y = delta_y * domain.dx[1]
        scaled_dx = (x1 - x) * domain.dx[0]
        return np.round((scaled_y * oblique_factor + np.sqrt(scaled_y*scaled_y + scaled_dx * scaled_dx)) / (c  * time_axis.dt)).astype(int).tolist()

    output_data_t = np.asarray(output_data)

    transducer_x_start = positions[0][0]
    transducer_x_end = positions[0][-1]
    transducer_y = positions[1][0]
    def compute_signal(pt_x, pt_y):
        delta_y = abs(transducer_y - pt_y)
        delays = compute_time_delays_for_point(positions[0], pt_x, delta_y)
        signal = np.zeros_like(output_data_t[:, 0])
        slanted_x_coord = int(pt_x - slope * delta_y)
        if slanted_x_coord > transducer_x_end or slanted_x_coord < transducer_x_start:
            return 0.0
        for i, delta in enumerate(delays):
            if abs(positions[0][i] - (pt_x - slope * delta_y)) < delta_y * abs(positions[0][1] - positions[0][0]):
                signal[:-delta] += output_data_t[delta:, i]
        return (np.dot(signal, s_stack_t[(slanted_x_coord - transducer_x_start) // abs(positions[0][1] - positions[0][0])]) * time_axis.dt).item()

    receive = np.array([[compute_signal(i, j) for j in range(0, domain.N[1])] for i in range(0, domain.N[0])])
    return receive

def get_receive_beamforming_medium_specific(domain, medium, time_axis, positions, output_data, signal, carrier_signal, signal_delay, c0=1500):
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
    # slope is in pixel space
    # TODO(vincent): assumes spacing of 1 between transducers
    s_stack_t = np.asarray(jnp.vstack([signal[i] for i in range(positions.shape[1])]))
    s_stack_t = s_stack_t / np.max(s_stack_t)
    
    def compute_single_segment_delay(x_1, y_1, x_2, y_2):
        # a weak estimate for the time to get from (x_1,y_1) to (x_2,y_2), assumes ray is relatively vertical
        sound_reciprocal_sums = 0.0
        for y in range(y_1, y_2):
            # TODO(vincent): can vectorize this with np
            frac_x = (x_1 * (y_2 - y) + x_2 * (y - y_1)) / (y_2 - y_1)
            int_x, partial_x = int(frac_x), frac_x - int(frac_x)
            sound_reciprocal_sums += (1 - partial_x) / medium.sound_speed.params[int_x,y,0] + partial_x / medium.sound_speed.params[int_x+1,y,0]
        ratio = (x_2 -x_1) / (y_2 - y_1)
        return ((ratio * ratio + 1)**0.5) * sound_reciprocal_sums * domain.dx[0] / time_axis.dt
    
    def compute_many_delays(x_1, y_1, xs, y_2):
        # TODO(vincent): this is inefficient, precompute it
        sound_reciprocal_sums = np.zeros(xs.shape)
        for y in range(y_1, y_2):
            x_coords = np.floor((x_1 * (y_2 - y) + xs * (y - y_1)) / (y_2 - y_1)).astype(int)
            sound_reciprocal_sums += 1/medium.sound_speed.params[x_coords,y,0]
        ratios = (xs - x_1) / (y_2 - y_1)
        return np.sqrt(ratios * ratios + 1) * domain.dx[0] * sound_reciprocal_sums / time_axis.dt

    def compute_time_delays_for_point(x1: np.ndarray, x: int, transducer_y: int, pt_y: int, c: float = c0):
        first_delay = compute_single_segment_delay(x, pt_y, int(x - (transducer_y - pt_y) * slope), transducer_y)
        second_delay = compute_many_delays(x, pt_y, x1, transducer_y)
        return np.round(first_delay + second_delay).astype(int).tolist()

    output_data_t = np.asarray(output_data)

    transducer_x_start = positions[0][0]
    transducer_x_end = positions[0][-1]
    transducer_y = positions[1][0]
    def compute_signal(pt_x, pt_y):
        print(pt_x, pt_y)
        if transducer_y <= pt_y:
            return 0.0
        delta_y = transducer_y - pt_y
        signal = np.zeros_like(output_data_t[:, 0])
        slanted_x_coord = int(pt_x - slope * delta_y)
        if slanted_x_coord > transducer_x_end or slanted_x_coord < transducer_x_start:
            return 0.0
        delays = compute_time_delays_for_point(positions[0], pt_x, transducer_y, pt_y)
        for i, delta in enumerate(delays):
            if delta > 0 and abs(positions[0][i] - (pt_x - slope * delta_y)) < delta_y * abs(positions[0][1] - positions[0][0]):
                signal[:-delta] += output_data_t[delta:, i]
        return (np.dot(signal, s_stack_t[(slanted_x_coord - transducer_x_start) // abs(positions[0][1] - positions[0][0])]) * time_axis.dt).item()

    print("hello")
    receive = np.array([[compute_signal(i, j) for j in range(0, domain.N[1])] for i in range(0, domain.N[0])])
    return receive