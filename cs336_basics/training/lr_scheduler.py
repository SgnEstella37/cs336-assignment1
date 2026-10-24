import math


def learning_rate_scheduling(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int):
    if t < T_w:
        return (t / T_w) * alpha_max

    if t > T_c:
        return alpha_min

    else:
        return alpha_min + (1 / 2) * (1 + math.cos(math.pi * (t - T_w) / (T_c - T_w))) * (alpha_max - alpha_min)
