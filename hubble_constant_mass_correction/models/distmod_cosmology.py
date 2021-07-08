from astropy.cosmology import FlatLambdaCDM


def get_mu_th(H0, Om0, z):
    return FlatLambdaCDM(H0=H0, Om0=Om0).distmod(z).value
