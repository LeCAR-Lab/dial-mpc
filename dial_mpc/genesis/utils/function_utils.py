import torch

def get_foot_step(duty_ratio, cadence, amplitude, phases, time):
    """
    Compute the foot step height.
    Args:
        amplitude: The height of the step.
        cadence: The cadence of the step (per second).
        duty_ratio: The duty ratio of the step (% on the ground).
        phases: The phase of the step. Warps around 1. (N-dim where N is the number of legs)
        time: The time of the step.
    """

    def step_height(t, footphase, duty_ratio):
        angle = (t + torch.pi - footphase) % (2 * torch.pi) - torch.pi
        angle = torch.where(duty_ratio < 1, angle * 0.5 / (1 - duty_ratio), angle)
        clipped_angle = torch.clip(angle, -torch.pi / 2, torch.pi / 2)
        value = torch.where(duty_ratio < 1, torch.cos(clipped_angle), 0)
        final_value = torch.where(torch.abs(value) >= 1e-6, torch.abs(value), 0.0)
        return final_value

    h_steps = amplitude * torch.vmap(step_height, in_dims=(None, 0, None))(
        time * 2 * torch.pi * cadence + torch.pi,
        2 * torch.pi * phases,
        duty_ratio,
    )
    return h_steps