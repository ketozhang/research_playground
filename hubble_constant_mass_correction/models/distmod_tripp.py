def get_mu_tripp(mag, stretch, color, alpha, beta, M_B):
    return mag - (M_B - alpha * stretch + beta * color)
