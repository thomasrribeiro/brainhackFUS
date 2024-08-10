import torch
import numpy as np
import jax.numpy as jnp

def compute_torch_signal(pt_x, pt_y, output_data, domain, nelements, transducer_depth, transducer_x_start, ss, delay_s, time_axis, positions, c: float = 1500):
    
    N = domain.N
    dx = domain.dx

    s_stack_t = torch.from_numpy(np.asarray(jnp.vstack([ss[i] for i in range(positions.shape[1])]))).to("cuda:0")
    s_stack_t /= torch.max(s_stack_t).item()

    output_data_t = torch.from_numpy(np.asarray(output_data)).to("cuda:0")

    element_y = (N[1] - transducer_depth)
    delta_y = abs(element_y - pt_y)
    
    slope = delay_s * time_axis.dt * c / dx[0].item()
    oblique_factor = (slope*slope+1)**0.5
    scaled_y = delta_y * dx[1]
    scaled_dx = (positions[0] - pt_x) * dx[0]
    delays = np.round((scaled_y * oblique_factor + np.sqrt(scaled_y*scaled_y + scaled_dx * scaled_dx)) / (c  * time_axis.dt)).astype(int).tolist()
    
    signal = torch.tensor([0.0]).to("cuda:0")
    slanted_x_coord = int(pt_x - slope * delta_y)
    if slanted_x_coord > transducer_x_start + nelements - 1 or slanted_x_coord < transducer_x_start:
        return np.array([0.0])
    for i in range(len(delays)):
        if abs(positions[0][i] - (pt_x - slope * delta_y)) < delta_y:
            delta = delays[i]
            signal += torch.dot(s_stack_t[slanted_x_coord - transducer_x_start, :-delta], output_data_t[delta:, i])
    return signal * time_axis.dt