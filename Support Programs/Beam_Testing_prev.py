from matplotlib.colors import Normalize
import numpy as np
from numpy import sin, cos, pi, arctan2, sqrt
import matplotlib.pyplot as plt
import cmath
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy.special import j0, j1, jvp, jv

c = 3e8


def gaussian_E_x(x, y, z, E0, wavelength, w0):
    x = x * 1 / w0
    y = y * 1 / w0
    z = (
        z * 1 / ((2 * np.pi / wavelength) * w0 ** 2)
    )  # probably not the best way to do it but can change later
    s = 1 / (2 * np.pi) * wavelength / w0

    E_x = (
        E0
        * (
            1
            + s ** 2
            * (
                -(x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 2
                + 1j * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 3
                - 2 * 1 / (1j + 2 * z) ** 2 * x ** 2
            )
            + s ** 4
            * (
                2 * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 4
                - 3j * (x ** 2 + y ** 2) ** 3 * 1 / (1j + 2 * z) ** 5
                - 0.5 * (x ** 2 + y ** 2) ** 4 * 1 / (1j + 2 * z) ** 6
                + (
                    8 * (x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 4
                    - 2j * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 5
                )
                * x ** 2
            )
        )
        * 1j
        / (1j + 2 * z)
        * cmath.exp(-1j * (x ** 2 + y ** 2) / (1j + 2 * z))
        * cmath.exp(-1j * z / s ** 2)
    )

    return E_x


def gaussian_dE_xdx(x, y, z, E0, wavelength, w0):
    x = x * 1 / w0
    y = y * 1 / w0
    z = (
        z * 1 / ((2 * np.pi / wavelength) * w0 ** 2)
    )  # probably not the best way to do it but can change later
    s = 1 / (2 * np.pi) * wavelength / w0

    gaussian_dE_xdx_scaled = E0 * (
        (
            s ** 2
            * (
                -2 * x * 1 / (1j + 2 * z) ** 2
                + 1j * (x ** 2 + y ** 2) * 4 * x * 1 / (1j + 2 * z) ** 3
                - 4 * 1 / (1j + 2 * z) ** 2 * x
            )
            + s ** 4
            * (
                8 * x * (x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 4
                - 18j * x * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 5
                - 4 * x * (x ** 2 + y ** 2) ** 3 * 1 / (1j + 2 * z) ** 6
                + (
                    16 * x * 1 / (1j + 2 * z) ** 4
                    - 8j * x * (x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 5
                )
                * x ** 2
                + 2
                * x
                * (
                    8 * (x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 4
                    - 2j * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 5
                )
            )
        )
        * 1j
        / (1j + 2 * z)
        * cmath.exp(-1j * (x ** 2 + y ** 2) / (1j + 2 * z))
        * cmath.exp(-1j * z / s ** 2)
    ) + E0 * (
        1j
        / (1j + 2 * z)
        * cmath.exp(-1j * (x ** 2 + y ** 2) / (1j + 2 * z))
        * cmath.exp(-1j * z / s ** 2)
        * -1j
        * 2
        * x
        / (1j + 2 * z)
        * (
            1
            + s ** 2
            * (
                -(x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 2
                + 1j * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 3
                - 2 * 1 / (1j + 2 * z) ** 2 * x ** 2
            )
            + s ** 4
            * (
                2 * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 4
                - 3j * (x ** 2 + y ** 2) ** 3 * 1 / (1j + 2 * z) ** 5
                - 0.5 * (x ** 2 + y ** 2) ** 4 * 1 / (1j + 2 * z) ** 6
                + (
                    8 * (x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 4
                    - 2j * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 5
                )
                * x ** 2
            )
        )
    )

    # Could shorten by making the variables beforehand and adding them in
    gaussian_dE_xdx = gaussian_dE_xdx_scaled * 1 / w0
    return gaussian_dE_xdx


def gaussian_dE_xdy(x, y, z, E0, wavelength, w0):
    x = x * 1 / w0
    y = y * 1 / w0
    z = (
        z * 1 / ((2 * np.pi / wavelength) * w0 ** 2)
    )  # probably not the best way to do it but can change later
    s = 1 / (2 * np.pi) * wavelength / w0

    gaussian_dE_xdy_scaled = E0 * (
        (
            s ** 2
            * (
                -2 * y * 1 / (1j + 2 * z) ** 2
                + 1j * (x ** 2 + y ** 2) * 4 * y * 1 / (1j + 2 * z) ** 3
            )
            + s ** 4
            * (
                8 * y * (x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 4
                - 18j * y * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 5
                - 4 * y * (x ** 2 + y ** 2) ** 3 * 1 / (1j + 2 * z) ** 6
                + (
                    16 * y * 1 / (1j + 2 * z) ** 4
                    - 8j * y * (x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 5
                )
                * x ** 2
            )
        )
        * 1j
        / (1j + 2 * z)
        * cmath.exp(-1j * (x ** 2 + y ** 2) / (1j + 2 * z))
        * cmath.exp(-1j * z / s ** 2)
    ) + E0 * (
        1j
        / (1j + 2 * z)
        * cmath.exp(-1j * (x ** 2 + y ** 2) / (1j + 2 * z))
        * cmath.exp(-1j * z / s ** 2)
        * -1j
        * 2
        * y
        / (1j + 2 * z)
        * (
            1
            + s ** 2
            * (
                -(x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 2
                + 1j * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 3
                - 2 * 1 / (1j + 2 * z) ** 2 * x ** 2
            )
            + s ** 4
            * (
                2 * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 4
                - 3j * (x ** 2 + y ** 2) ** 3 * 1 / (1j + 2 * z) ** 5
                - 0.5 * (x ** 2 + y ** 2) ** 4 * 1 / (1j + 2 * z) ** 6
                + (
                    8 * (x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 4
                    - 2j * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 5
                )
                * x ** 2
            )
        )
    )

    # Could shorten by making the variables beforehand and adding them in
    gaussian_dE_xdy = gaussian_dE_xdy_scaled * 1 / w0
    return gaussian_dE_xdy


def gaussian_dE_xdz(x, y, z, E0, wavelength, w0):
    x = x * 1 / w0
    y = y * 1 / w0
    z = (
        z * 1 / ((2 * np.pi / wavelength) * w0 ** 2)
    )  # probably not the best way to do it but can change later
    s = 1 / (2 * np.pi) * wavelength / w0

    gaussian_dE_xdz_scaled = E0 * (
        s ** 2
        * (
            -(x ** 2 + y ** 2) * -4 / (1j + 2 * z) ** 3
            + 1j * (x ** 2 + y ** 2) ** 2 * -6 / (1j + 2 * z) ** 4
            - 2 * -4 / (1j + 2 * z) ** 3 * x ** 2
        )
        + s ** 4
        * (
            2 * (x ** 2 + y ** 2) ** 2 * -8 / (1j + 2 * z) ** 5
            - 3j * (x ** 2 + y ** 2) ** 3 * -10 / (1j + 2 * z) ** 6
            - 0.5 * (x ** 2 + y ** 2) ** 4 * -12 / (1j + 2 * z) ** 7
            + (
                8 * (x ** 2 + y ** 2) * -8 / (1j + 2 * z) ** 8
                - 2j * (x ** 2 + y ** 2) ** 2 * -10 / (1j + 2 * z) ** 6
            )
            * x ** 2
        )
    ) * 1j / (1j + 2 * z) * cmath.exp(
        -1j * (x ** 2 + y ** 2) / (1j + 2 * z)
    ) * cmath.exp(
        -1j * z / s ** 2
    ) + E0 * (
        1
        + s ** 2
        * (
            -(x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 2
            + 1j * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 3
            - 2 * 1 / (1j + 2 * z) ** 2 * x ** 2
        )
        + s ** 4
        * (
            2 * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 4
            - 3j * (x ** 2 + y ** 2) ** 3 * 1 / (1j + 2 * z) ** 5
            - 0.5 * (x ** 2 + y ** 2) ** 4 * 1 / (1j + 2 * z) ** 6
            + (
                8 * (x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 4
                - 2j * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 5
            )
            * x ** 2
        )
    ) * (
        1j
        / (1j + 2 * z)
        * cmath.exp(-1j * (x ** 2 + y ** 2) / (1j + 2 * z))
        * -1j
        / (s ** 2)
        * cmath.exp(-1j * z / (s ** 2))
        + cmath.exp(-1j * z / (s ** 2))
        * (
            (
                1j / (1j + 2 * z) * 2j * (x ** 2 + y ** 2) / (1j + 2 * z) ** 2
                - 2j / (1j + 2 * z) ** 2
            )
            * cmath.exp(-1j * (x ** 2 + y ** 2) / (1j + 2 * z))
        )
    )
    # Could shorten by making the variables beforehand and adding them in
    gaussian_dE_xdz = gaussian_dE_xdz_scaled * 1 / (2 * (np.pi / wavelength) * w0 ** 2)
    return gaussian_dE_xdz


def gaussian_E_y(x, y, z, E0, wavelength, w0):
    x = x * 1 / w0
    y = y * 1 / w0
    z = z * 1 / ((2 * np.pi / wavelength) * w0 ** 2)
    s = 1 / (2 * np.pi) * wavelength / w0

    E_y = (
        E0
        * (
            s ** 2 * (-2 * 1 / (1j + 2 * z) ** 2 * x * y)
            + s ** 4
            * (
                8 * (x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 4
                - 2j * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 5
            )
            * x
            * y
        )
        * 1j
        / (1j + 2 * z)
        * cmath.exp(-1j * (x ** 2 + y ** 2) / (1j + 2 * z))
        * cmath.exp(-1j * z / s ** 2)
    )

    return E_y


def gaussian_dE_ydx(x, y, z, E0, wavelength, w0):
    x = x * 1 / w0
    y = y * 1 / w0
    z = z * 1 / ((2 * np.pi / wavelength) * w0 ** 2)
    s = 1 / (2 * np.pi) * (wavelength / w0)

    gaussian_dE_ydx_scaled = (
        E0
        * (
            s ** 2 * y * -2 / (-1 + 2j * z + 4 * z ** 2)
            + s ** 4
            * (
                16 * x * 1 / (1j + 2 * z) ** 4
                - 8j * x * (x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 5
            )
            * x
            * y
            + s ** 4
            * (
                8 * (x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 4
                - 2j * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 5
            )
            * y
        )
        * 1j
        / (1j + 2 * z)
        * cmath.exp(-1j * (x ** 2 + y ** 2) / (1j + 2 * z))
        * cmath.exp(-1j * z / s ** 2)
    ) + E0 * (
        s ** 2 * (-2 * 1 / (1j + 2 * z) ** 2 * x * y)
        + s ** 4
        * (
            8 * (x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 4
            - 2j * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 5
        )
        * x
        * y
    ) * 1j / (
        1j + 2 * z
    ) * cmath.exp(
        -1j * (x ** 2 + y ** 2) / (1j + 2 * z)
    ) * cmath.exp(
        -1j * z / s ** 2
    ) * -2j * x / (
        1j + 2 * z
    )
    gaussian_dE_ydx = gaussian_dE_ydx_scaled * 1 / w0
    return gaussian_dE_ydx


def gaussian_dE_ydy(x, y, z, E0, wavelength, w0):
    x = x * 1 / w0
    y = y * 1 / w0
    z = z * 1 / ((2 * np.pi / wavelength) * w0 ** 2)
    s = 1 / (2 * np.pi) * wavelength / w0

    gaussian_dE_ydy_scaled = (
        E0
        * (
            s ** 2 * (-2 * 1 / (1j + 2 * z) ** 2 * x)
            + s ** 4
            * (
                16 * y * 1 / (1j + 2 * z) ** 4
                - 8j * y * (x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 5
            )
            * x
            * y
            + s ** 4
            * (
                8 * (x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 4
                - 2j * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 5
            )
            * x
        )
        * 1j
        / (1j + 2 * z)
        * cmath.exp(-1j * (x ** 2 + y ** 2) / (1j + 2 * z))
        * cmath.exp(-1j * z / s ** 2)
    ) + E0 * (
        s ** 2 * (-2 * 1 / (1j + 2 * z) ** 2 * x * y)
        + s ** 4
        * (
            8 * (x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 4
            - 2j * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 5
        )
        * x
        * y
    ) * 1j / (
        1j + 2 * z
    ) * cmath.exp(
        -1j * (x ** 2 + y ** 2) / (1j + 2 * z)
    ) * cmath.exp(
        -1j * z / s ** 2
    ) * -2j * y / (
        1j + 2 * z
    )

    gaussian_dE_ydy = gaussian_dE_ydy_scaled * 1 / w0
    return gaussian_dE_ydy


def gaussian_dE_ydz(x, y, z, E0, wavelength, w0):
    x = x * 1 / w0
    y = y * 1 / w0
    z = z * 1 / ((2 * np.pi / wavelength) * w0 ** 2)
    s = 1 / (2 * np.pi) * wavelength / w0

    gaussian_dE_ydz_scaled = (
        E0
        * (
            s ** 2 * (8 * 1 / (1j + 2 * z) ** 3 * x * y)
            + s ** 4
            * (
                8 * (x ** 2 + y ** 2) * -8 / (1j + 2 * z) ** 5
                - 2j * (x ** 2 + y ** 2) ** 2 * -10 / (1j + 2 * z) ** 6
            )
            * x
            * y
        )
        * 1j
        / (1j + 2 * z)
        * cmath.exp(-1j * (x ** 2 + y ** 2) / (1j + 2 * z))
        * cmath.exp(-1j * z / s ** 2)
    ) + E0 * (
        s ** 2 * (-2 * 1 / (1j + 2 * z) ** 2 * x * y)
        + s ** 4
        * (
            8 * (x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 4
            - 2j * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 5
        )
        * x
        * y
    ) * (
        1j
        / (1j + 2 * z)
        * cmath.exp(-1j * (x ** 2 + y ** 2) / (1j + 2 * z))
        * -1j
        / (s ** 2)
        * cmath.exp(-1j * z / (s ** 2))
        + cmath.exp(-1j * z / (s ** 2))
        * (
            1j
            / (1j + 2 * z)
            * (
                2j
                * (x ** 2 + y ** 2)
                / (1j + 2 * z) ** 2
                * cmath.exp(-1j * (x ** 2 + y ** 2) / (1j + 2 * z))
            )
            - 2
            * 1j
            / (1j + 2 * z) ** 2
            * cmath.exp(-1j * (x ** 2 + y ** 2) / (1j + 2 * z))
        )
    )
    gaussian_dE_ydz = gaussian_dE_ydz_scaled * 1 / (2 * (np.pi / wavelength) * w0 ** 2)
    return gaussian_dE_ydz


def gaussian_E_z(x, y, z, E0, wavelength, w0):
    x = x * 1 / w0
    y = y * 1 / w0
    z = z * 1 / ((2 * np.pi / wavelength) * w0 ** 2)
    s = 1 / (2 * np.pi) * wavelength / w0

    E_z = E0 * (
        (
            s * (-2 * 1 / (1j + 2 * z) * x)
            + s ** 3
            * (
                (
                    6 * (x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 3
                    - 2j * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 4
                )
                * x
            )
            + s ** 5
            * (
                (
                    -20 * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 5
                    + 10j * (x ** 2 + y ** 2) ** 3 * 1 / (1j + 2 * z) ** 6
                    + (x ** 2 + y ** 2) ** 4 * 1 / (1j + 2 * z) ** 7
                )
                * x
            )
        )
        * 1j
        / (1j + 2 * z)
        * cmath.exp(-1j * (x ** 2 + y ** 2) / (1j + 2 * z))
        * cmath.exp(-1j * z / s ** 2)
    )

    return E_z


def gaussian_dE_zdx(x, y, z, E0, wavelength, w0):
    x = x * 1 / w0
    y = y * 1 / w0
    z = z * 1 / ((2 * np.pi / wavelength) * w0 ** 2)
    s = 1 / (2 * np.pi) * wavelength / w0

    gaussian_dE_zdx_scaled = E0 * (
        (
            s * (-2 * 1 / (1j + 2 * z))
            + s ** 3
            * (
                (
                    12 * x * 1 / (1j + 2 * z) ** 3
                    - 8j * x * (x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 4
                )
                * x
                + s ** 3
                * (
                    6 * (x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 3
                    - 2j * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 4
                )
            )
            + s ** 5
            * (
                (
                    -40 * x * (x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 5
                    + 60j * x * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 6
                    + 8 * x * (x ** 2 + y ** 2) ** 3 * 1 / (1j + 2 * z) ** 7
                )
                * x
            )
            + s ** 5
            * (
                -20 * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 5
                + 10j * (x ** 2 + y ** 2) ** 3 * 1 / (1j + 2 * z) ** 6
                + (x ** 2 + y ** 2) ** 4 * 1 / (1j + 2 * z) ** 7
            )
        )
        * 1j
        / (1j + 2 * z)
        * cmath.exp(-1j * (x ** 2 + y ** 2) / (1j + 2 * z))
        * cmath.exp(-1j * z / s ** 2)
    ) + E0 * (
        s * (-2 * 1 / (1j + 2 * z) * x)
        + s ** 3
        * (
            (
                6 * (x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 3
                - 2j * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 4
            )
            * x
        )
        + s ** 5
        * (
            (
                -20 * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 5
                + 10j * (x ** 2 + y ** 2) ** 3 * 1 / (1j + 2 * z) ** 6
                + (x ** 2 + y ** 2) ** 4 * 1 / (1j + 2 * z) ** 7
            )
            * x
        )
    ) * 1j / (
        1j + 2 * z
    ) * cmath.exp(
        -1j * (x ** 2 + y ** 2) / (1j + 2 * z)
    ) * cmath.exp(
        -1j * z / s ** 2
    ) * -2j * x / (
        1j + 2 * z
    )
    gaussian_dE_zdx = gaussian_dE_zdx_scaled * 1 / w0
    return gaussian_dE_zdx


def gaussian_dE_zdy(x, y, z, E0, wavelength, w0):
    x = x * 1 / w0
    y = y * 1 / w0
    z = z * 1 / ((2 * np.pi / wavelength) * w0 ** 2)
    s = 1 / (2 * np.pi) * wavelength / w0

    gaussian_dE_zdy_scaled = E0 * (
        (
            s ** 3
            * (
                12 * y * 1 / (1j + 2 * z) ** 3
                - 8j * y * (x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 4
            )
            * x
            + s ** 5
            * (
                (
                    -40 * y * (x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 5
                    + 60j * y * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 6
                    + 8 * y * (x ** 2 + y ** 2) ** 3 * 1 / (1j + 2 * z) ** 7
                )
                * x
            )
        )
        * 1j
        / (1j + 2 * z)
        * cmath.exp(-1j * (x ** 2 + y ** 2) / (1j + 2 * z))
        * cmath.exp(-1j * z / s ** 2)
    ) + E0 * (
        (
            s * (-2 * 1 / (1j + 2 * z) * x)
            + s ** 3
            * (
                (
                    6 * (x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 3
                    - 2j * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 4
                )
                * x
            )
            + s ** 5
            * (
                (
                    -20 * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 5
                    + 10j * (x ** 2 + y ** 2) ** 3 * 1 / (1j + 2 * z) ** 6
                    + (x ** 2 + y ** 2) ** 4 * 1 / (1j + 2 * z) ** 7
                )
                * x
            )
        )
        * 1j
        / (1j + 2 * z)
        * cmath.exp(-1j * (x ** 2 + y ** 2) / (1j + 2 * z))
        * cmath.exp(-1j * z / s ** 2)
        * -2j
        * y
        / (1j + 2 * z)
    )
    gaussian_dE_zdy = gaussian_dE_zdy_scaled * 1 / w0
    return gaussian_dE_zdy


def gaussian_dE_zdz(x, y, z, E0, wavelength, w0):
    x = x * 1 / w0
    y = y * 1 / w0
    z = z * 1 / ((2 * np.pi / wavelength) * w0 ** 2)
    s = 1 / (2 * np.pi) * wavelength / w0

    gaussian_dE_zdz_scaled = E0 * (
        s * (4 * 1 / (1j + 2 * z) ** 2 * x)
        + s ** 3
        * (
            (
                -36 * (x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 4
                + 16j * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 5
            )
            * x
        )
        + s ** 5
        * (
            (
                200 * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 6
                - 120j * (x ** 2 + y ** 2) ** 3 * 1 / (1j + 2 * z) ** 7
                - 14 * (x ** 2 + y ** 2) ** 4 * 1 / (1j + 2 * z) ** 8
            )
            * x
        )
    ) * 1j / (1j + 2 * z) * cmath.exp(
        -1j * (x ** 2 + y ** 2) / (1j + 2 * z)
    ) * cmath.exp(
        -1j * z / s ** 2
    ) + E0 * (
        s * (-2 * 1 / (1j + 2 * z) * x)
        + s ** 3
        * (
            (
                6 * (x ** 2 + y ** 2) * 1 / (1j + 2 * z) ** 3
                - 2j * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 4
            )
            * x
        )
        + s ** 5
        * (
            (
                -20 * (x ** 2 + y ** 2) ** 2 * 1 / (1j + 2 * z) ** 5
                + 10j * (x ** 2 + y ** 2) ** 3 * 1 / (1j + 2 * z) ** 6
                + (x ** 2 + y ** 2) ** 4 * 1 / (1j + 2 * z) ** 7
            )
            * x
        )
    ) * (
        1j
        / (1j + 2 * z)
        * cmath.exp(-1j * (x ** 2 + y ** 2) / (1j + 2 * z))
        * -1j
        / (s ** 2)
        * cmath.exp(-1j * z / (s ** 2))
        + cmath.exp(-1j * z / (s ** 2))
        * (
            (
                1j / (1j + 2 * z) * 2j * (x ** 2 + y ** 2) / (1j + 2 * z) ** 2
                - 2j / (1j + 2 * z) ** 2
            )
            * cmath.exp(-1j * (x ** 2 + y ** 2) / (1j + 2 * z))
        )
    )
    gaussian_dE_zdz = gaussian_dE_zdz_scaled * 1 / (2 * (np.pi / wavelength) * w0 ** 2)
    return gaussian_dE_zdz


def bessel_constants(alpha_by_k, wavelength, x, y):
    epsilon = 1
    k = 2 * np.pi / wavelength * np.sqrt(epsilon)
    alpha = alpha_by_k * k
    beta = np.sqrt(k ** 2 - alpha ** 2)
    b = 1 + beta / k
    r = np.sqrt(x ** 2 + y ** 2)

    return alpha, beta, k, r, b


def bessel_E_x(x, y, z, E0, wavelength, alpha):
    alpha, beta, k, r, b = bessel_constants(alpha, wavelength, x, y)
    Ex = (
        0.5
        * E0
        * (
            (b - alpha ** 2 * x ** 2 / (k ** 2 * r ** 2)) * j0(alpha * r)
            - alpha * (r ** 2 - 2 * x ** 2) / (k ** 2 * r ** 3) * j1(alpha * r)
        )
        * cmath.exp(1j * beta * z)
    )

    return Ex


def bessel_dE_xdx(x, y, z, E0, wavelength, alpha_by_k):
    alpha, beta, k, r, b = bessel_constants(alpha_by_k, wavelength, x, y)

    dE_xdx = (
        0.5
        * E0
        * (
            -(alpha ** 2)
            / k ** 2
            * (2 * x / (x ** 2 + y ** 2) - 2 * x ** 3 / (x ** 2 + y ** 2) ** 2)
            * j0(alpha * r)
            + (b - alpha ** 2 * x ** 2 / (k ** 2 * r ** 2))
            * jvp(0, alpha * r, 1)
            * x
            * alpha
            * (x ** 2 + y ** 2) ** (-1 / 2)
            - (
                alpha
                / k ** 2
                * (
                    -2 * x * (x ** 2 + y ** 2) ** (-3 / 2)
                    + 3 * x ** 3 * (x ** 2 + y ** 2) ** (-5 / 2)
                    + y ** 2 * (-3 / 2) * 2 * x * (x ** 2 + y ** 2) ** (-5 / 2)
                )
            )
            * j1(alpha * r)
            - alpha
            * (r ** 2 - 2 * x ** 2)
            / (k ** 2 * r ** 3)
            * jvp(1, alpha * r, 1)
            * x
            * (x ** 2 + y ** 2) ** (-1 / 2)
            * alpha
        )
        * cmath.exp(1j * beta * z)
    )

    return dE_xdx


def bessel_dE_xdy(x, y, z, E0, wavelength, alpha):
    alpha, beta, k, r, b = bessel_constants(alpha, wavelength, x, y)

    dE_xdy = (
        0.5
        * E0
        * (
            -(alpha ** 2)
            / k ** 2
            * x ** 2
            * 2
            * y
            * (x ** 2 + y ** 2) ** (-2)
            * j0(alpha * r)
            + (b - alpha ** 2 * x ** 2 / (k ** 2 * r ** 2))
            * jvp(0, alpha * r, 1)
            * y
            * alpha
            * (x ** 2 + y ** 2) ** (-1 / 2)
            - (
                alpha
                / k ** 2
                * (
                    2 * y * (x ** 2 + y ** 2) ** (-3 / 2)
                    - 3 * y ** 3 * (x ** 2 + y ** 2) ** (-5 / 2)
                    + x ** 2 * (3 / 2) * 2 * y * (x ** 2 + y ** 2) ** (-5 / 2)
                )
            )
            * j1(alpha * r)
            - alpha
            * (r ** 2 - 2 * x ** 2)
            / (k ** 2 * r ** 3)
            * jvp(1, alpha * r, 1)
            * y
            * (x ** 2 + y ** 2) ** (-1 / 2)
            * alpha
        )
        * cmath.exp(1j * beta * z)
    )

    return dE_xdy


def bessel_dE_xdz(x, y, z, E0, wavelength, alpha):
    alpha, beta, k, r, b = bessel_constants(alpha, wavelength, x, y)
    dExdz = (
        0.5
        * E0
        * (
            (b - alpha ** 2 * x ** 2 / (k ** 2 * r ** 2)) * j0(alpha * r)
            - alpha * (r ** 2 - 2 * x ** 2) / (k ** 2 * r ** 3) * j1(alpha * r)
        )
        * cmath.exp(1j * beta * z)
        * 1j
        * beta
    )

    return dExdz


def bessel_E_y(x, y, z, E0, wavelength, alpha_by_k):
    alpha, beta, k, r, b = bessel_constants(alpha_by_k, wavelength, x, y)
    Ey = (
        0.5
        * E0
        * (
            2 * alpha * x * y / (k ** 2 * r ** 3) * j1(alpha * r)
            - alpha ** 2 * x * y / (k ** 2 * r ** 2) * j0(alpha * r)
        )
        * cmath.exp(1j * beta * z)
    )

    return Ey


def bessel_dE_ydx(x, y, z, E0, wavelength, alpha):
    alpha, beta, k, r, b = bessel_constants(alpha, wavelength, x, y)
    dEydx = (
        0.5
        * E0
        * (
            (
                2 * alpha * y / (k ** 2 * r ** 3)
                + 2 * alpha * x * y * -3 * x / (k ** 2 * r ** 5)
            )
            * j1(alpha * r)
            + (2 * alpha * x * y / (k ** 2 * r ** 3))
            * jvp(1, alpha * r, 1)
            * x
            * alpha
            * (x ** 2 + y ** 2) ** (-1 / 2)
            - (
                alpha ** 2 / k ** 2 * y * (x ** 2 + y ** 2) ** (-1)
                + alpha ** 2 / k ** 2 * x * y * (x ** 2 + y ** 2) ** (-2) * 2 * x
            )
            * j0(alpha * r)
            - (alpha ** 2 * x * y / (k ** 2 * r ** 2))
            * jvp(0, alpha * r, 1)
            * x
            * alpha
            * (x ** 2 + y ** 2) ** (-1 / 2)
        )
        * cmath.exp(1j * beta * z)
    )

    return dEydx


def bessel_dE_ydy(x, y, z, E0, wavelength, alpha):
    alpha, beta, k, r, b = bessel_constants(alpha, wavelength, x, y)
    dEydy = (
        0.5
        * E0
        * (
            (
                2 * alpha * 1 / k ** 2 * x * (x ** 2 + y ** 2) ** (-3 / 2)
                + 2 * alpha / k ** 2 * x * y * -3 * y * (x ** 2 + y ** 2) ** (-5 / 2)
            )
            * j1(alpha * r)
            + (2 * alpha * x * y / (k ** 2 * r ** 3))
            * jvp(1, alpha * r, 1)
            * y
            * alpha
            * (x ** 2 + y ** 2) ** (-1 / 2)
            - (
                alpha ** 2 / k ** 2 * x * (x ** 2 + y ** 2) ** (-1)
                + alpha ** 2 / k ** 2 * x * y * (x ** 2 + y ** 2) ** (-2) * 2 * y
            )
            * j0(alpha * r)
            - (alpha ** 2 * x * y / (k ** 2 * r ** 2))
            * jvp(0, alpha * r, 1)
            * y
            * alpha
            * (x ** 2 + y ** 2) ** (-1 / 2)
        )
        * cmath.exp(1j * beta * z)
    )

    return dEydy


def bessel_dE_ydz(x, y, z, E0, wavelength, alpha):
    alpha, beta, k, r, b = bessel_constants(alpha, wavelength, x, y)
    dEydz = (
        0.5
        * E0
        * (
            2 * alpha * x * y / (k ** 2 * r ** 3) * j1(alpha * r)
            - alpha ** 2 * x * y / (k ** 2 * r ** 2) * j0(alpha * r)
        )
        * cmath.exp(1j * beta * z)
        * 1j
        * beta
    )

    return dEydz


def bessel_E_z(x, y, z, E0, wavelength, alpha):
    alpha, beta, k, r, b = bessel_constants(alpha, wavelength, x, y)

    Ez = (
        0.5
        * E0
        * (-1j * alpha * b * x / (k * r) * j1(alpha * r))
        * cmath.exp(1j * beta * z)
    )

    return Ez


def bessel_dE_zdx(x, y, z, E0, wavelength, alpha):
    alpha, beta, k, r, b = bessel_constants(alpha, wavelength, x, y)

    dEzdx = (
        0.5
        * E0
        * (
            -1j
            * alpha
            * b
            / k
            * ((x ** 2 + y ** 2) ** (-1 / 2) - x ** 2 * (x ** 2 + y ** 2) ** (-3 / 2))
            * j1(alpha * r)
            - 1j
            * alpha
            * b
            * x
            / (k * r)
            * jvp(1, alpha * r, 1)
            * x
            * alpha
            * (x ** 2 + y ** 2) ** (-1 / 2)
        )
        * cmath.exp(1j * beta * z)
    )

    return dEzdx


def bessel_dE_zdy(x, y, z, E0, wavelength, alpha):
    alpha, beta, k, r, b = bessel_constants(alpha, wavelength, x, y)

    dEzdy = (
        0.5
        * E0
        * (
            (1j * alpha * b * y * x / k * r ** 3) * j1(alpha * r)
            - 1j
            * alpha
            * b
            * x
            / (k * r)
            * jvp(1, alpha * r, 1)
            * y
            * alpha
            * (x ** 2 + y ** 2) ** (-1 / 2)
        )
        * cmath.exp(1j * beta * z)
    )

    return dEzdy


def bessel_dE_zdz(x, y, z, E0, wavelength, alpha):
    alpha, beta, k, r, b = bessel_constants(alpha, wavelength, x, y)

    dEzdz = (
        0.5
        * E0
        * (-1j * alpha * b * x / (k * r) * j1(alpha * r))
        * cmath.exp(1j * beta * z)
        * 1j
        * beta
    )

    return dEzdz


def volke_bessel(x, y, z, E0):
    l = 1
    r = np.sqrt(x ** 2 + y ** 2)
    if x >= 0:
        phi = np.arctan(y / x)
    elif x < 0:
        phi = -np.arcsin(y / (r)) + pi
    kz = k
    kt = k / 10
    a = jones_vector[0]
    b = jones_vector[1]
    Ex = E0 * cmath.exp(1j * (kz * z + l * phi)) * a * jv(l, kt * r)
    Ey = E0 * cmath.exp(1j * (kz * z + l * phi)) * b * jv(l, kt * r)
    Ez = (
        E0
        * cmath.exp(1j * (kz * z + l * phi))
        * (
            (a + b * 1j) * cmath.exp(-1j * phi) * jv(l - 1, kt * r)
            - (a - b * 1j) * cmath.exp(1j * phi) * jv(l + 1, kt * r)
        )
        * 1j
        / 2
        * (kt / kz)
    )
    return Ex, Ey, Ez


def general_bessel_constants(wavelength, x, y, kt_by_kz):
    k = (2 * np.pi) / wavelength
    r = np.sqrt(x ** 2 + y ** 2)
#    kz = k
    kz = k/np.sqrt(kt_by_kz**2 + 1)
    kt = kt_by_kz * kz
    a = jones_vector[0]
    b = jones_vector[1]
    phi = arctan2(y, x)

    return kz, kt, r, a, b, phi


# use general bessel instead of first bessel
def first_bessel_E_x(x, y, z, E0, wavelength):
    kz, kt, r, a, b, phi = general_bessel_constants(wavelength, x, y, kt_by_kz)

    Ex = E0 * cmath.exp(1j * kz * z) * ((x + 1j * y) / (r)) * a * j1(kt * r)

    return Ex


def first_bessel_dE_xdx(x, y, z, E0, wavelength):
    kz, kt, r, a, b, phi = general_bessel_constants(wavelength, x, y, kt_by_kz)

    dExdx = (
        a
        * E0
        * cmath.exp(1j * kz * z)
        * (
            (
                (x ** 2 + y ** 2) ** (-1 / 2)
                - x * (x + 1j * y) * (x ** 2 + y ** 2) ** (-3 / 2)
            )
            * j1(kt * r)
            + ((x + 1j * y) / (r))
            * jvp(1, kt * r, 1)
            * kt
            * x
            * (x ** 2 + y ** 2) ** (-1 / 2)
        )
    )

    return dExdx


def first_bessel_dE_xdy(x, y, z, E0, wavelength):
    kz, kt, r, a, b, phi = general_bessel_constants(wavelength, x, y, kt_by_kz)

    dExdy = (
        a
        * E0
        * cmath.exp(1j * kz * z)
        * (
            (
                1j * (x ** 2 + y ** 2) ** (-1 / 2)
                - y * (x + 1j * y) * (x ** 2 + y ** 2) ** (-3 / 2)
            )
            * j1(kt * r)
            + ((x + 1j * y) / (r))
            * jvp(1, kt * r, 1)
            * kt
            * y
            * (x ** 2 + y ** 2) ** (-1 / 2)
        )
    )

    return dExdy


def first_bessel_dE_xdz(x, y, z, E0, wavelength):
    kz, kt, r, a, b, phi = general_bessel_constants(wavelength, x, y, kt_by_kz)

    dExdz = (
        E0 * a * ((x + 1j * y) / (r)) * j1(kt * r) * 1j * kz * cmath.exp(1j * kz * z)
    )

    return dExdz


def first_bessel_E_y(x, y, z, E0, wavelength):
    kz, kt, r, a, b, phi = general_bessel_constants(wavelength, x, y, kt_by_kz)

    Ex = E0 * cmath.exp(1j * kz * z) * ((x + 1j * y) / (r)) * b * j1(kt * r)

    return Ex


def first_bessel_dE_ydx(x, y, z, E0, wavelength):
    kz, kt, r, a, b, phi = general_bessel_constants(wavelength, x, y, kt_by_kz)

    dExdx = (
        b
        * E0
        * cmath.exp(1j * kz * z)
        * (
            (
                (x ** 2 + y ** 2) ** (-1 / 2)
                - x * (x + 1j * y) * (x ** 2 + y ** 2) ** (-3 / 2)
            )
            * j1(kt * r)
            + ((x + 1j * y) / (r))
            * jvp(1, kt * r, 1)
            * kt
            * x
            * (x ** 2 + y ** 2) ** (-1 / 2)
        )
    )

    return dExdx


def first_bessel_dE_ydy(x, y, z, E0, wavelength):
    kz, kt, r, a, b, phi = general_bessel_constants(wavelength, x, y, kt_by_kz)

    dExdy = (
        b
        * E0
        * cmath.exp(1j * kz * z)
        * (
            (
                1j * (x ** 2 + y ** 2) ** (-1 / 2)
                - y * (x + 1j * y) * (x ** 2 + y ** 2) ** (-3 / 2)
            )
            * j1(kt * r)
            + ((x + 1j * y) / (r))
            * jvp(1, kt * r, 1)
            * kt
            * y
            * (x ** 2 + y ** 2) ** (-1 / 2)
        )
    )

    return dExdy


def first_bessel_dE_ydz(x, y, z, E0, wavelength):
    kz, kt, r, a, b, phi = general_bessel_constants(wavelength, x, y, kt_by_kz)

    dExdz = (
        E0 * b * ((x + 1j * y) / (r)) * j1(kt * r) * 1j * kz * cmath.exp(1j * kz * z)
    )

    return dExdz


def first_bessel_E_z(x, y, z, E0, wavelength):
    kz, kt, r, a, b, phi = general_bessel_constants(wavelength, x, y, kt_by_kz)

    Ez = (
        E0
        * cmath.exp(1j * kz * z)
        * 1j
        / 2
        * (kt / kz)
        * (
            (x + 1j * y)
            / r
            * (
                (a + 1j * b) * (x - 1j * y) / r * j0(kt * r)
                - (a - 1j * b) * (x + 1j * y) / r * jv(2, kt * r)
            )
        )
    )

    return Ez


def first_bessel_dE_zdx(x, y, z, E0, wavelength):
    kz, kt, r, a, b, phi = general_bessel_constants(wavelength, x, y, kt_by_kz)

    dEzdx = (
        E0
        * cmath.exp(1j * kz * z)
        * (
            (
                (x ** 2 + y ** 2) ** (-1 / 2)
                - x * (x + 1j * y) * (x ** 2 + y ** 2) ** (-3 / 2)
            )
            * (
                (a + 1j * b) * (x - 1j * y) / r * j0(kt * r)
                - (a - 1j * b) * (x + 1j * y) / r * jv(2, kt * r)
            )
            + (x + 1j * y)
            / r
            * (
                (a + 1j * b)
                * (
                    (
                        (x ** 2 + y ** 2) ** (-1 / 2)
                        - x * (x - 1j * y) * (x ** 2 + y ** 2) ** (-3 / 2)
                    )
                    * j0(kt * r)
                    + (x - 1j * y)
                    / r
                    * jvp(0, kt * r, 1)
                    * kt
                    * x
                    * (x ** 2 + y ** 2) ** (-1 / 2)
                )
                - (a - 1j * b)
                * (
                    (
                        (x ** 2 + y ** 2) ** (-1 / 2)
                        - x * (x + 1j * y) * (x ** 2 + y ** 2) ** (-3 / 2)
                    )
                    * jv(2, kt * r)
                    + (x + 1j * y)
                    / r
                    * jvp(2, kt * r, 1)
                    * kt
                    * x
                    * (x ** 2 + y ** 2) ** (-1 / 2)
                )
            )
        )
        * 1j
        / 2
        * (kt / kz)
    )

    return dEzdx


def first_bessel_dE_zdy(x, y, z, E0, wavelength):
    kz, kt, r, a, b, phi = general_bessel_constants(wavelength, x, y, kt_by_kz)

    dEzdy = (
        E0
        * cmath.exp(1j * kz * z)
        * (
            (
                (x ** 2 + y ** 2) ** (-1 / 2)
                - y * (x + 1j * y) * (x ** 2 + y ** 2) ** (-3 / 2)
            )
            * (
                (a + 1j * b) * (x - 1j * y) / r * j0(kt * r)
                - (a - 1j * b) * (x + 1j * y) / r * jv(2, kt * r)
            )
            + (x + 1j * y)
            / r
            * (
                (a + 1j * b)
                * (
                    (
                        (x ** 2 + y ** 2) ** (-1 / 2)
                        + y * (x - 1j * y) * (x ** 2 + y ** 2) ** (-3 / 2)
                    )
                    * j0(kt * r)
                    + (x - 1j * y)
                    / r
                    * jvp(0, kt * r, 1)
                    * kt
                    * y
                    * (x ** 2 + y ** 2) ** (-1 / 2)
                )
                - (a - 1j * b)
                * (
                    (
                        (x ** 2 + y ** 2) ** (-1 / 2)
                        - y * (x + 1j * y) * (x ** 2 + y ** 2) ** (-3 / 2)
                    )
                    * jv(2, kt * r)
                    + (x + 1j * y)
                    / r
                    * jvp(2, kt * r, 1)
                    * kt
                    * y
                    * (x ** 2 + y ** 2) ** (-1 / 2)
                )
            )
        )
        * 1j
        / 2
        * (kt / kz)
    )

    return dEzdy


def first_bessel_dE_zdz(x, y, z, E0, wavelength):
    kz, kt, r, a, b, phi = general_bessel_constants(wavelength, x, y, kt_by_kz)

    dEzdz = (
        E0
        * cmath.exp(1j * kz * z)
        * (
            (x + 1j * y)
            / r
            * (
                (a + 1j * b) * (x - 1j * y) / r * j0(kt * r)
                - (a - 1j * b) * (x + 1j * y) / r * jv(2, kt * r)
            )
            * 1j
            / 2
            * (kt / kz)
        )
        * 1j
        * kz
    )

    return dEzdz


def general_bessel_E_x(x, y, z, E0, wavelength, order):
    kz, kt, r, a, b, phi = general_bessel_constants(wavelength, x, y, kt_by_kz)
    l = order
    Ex = E0 * cmath.exp(1j * kz * z) * cmath.exp(1j * l * phi) * a * jv(l, kt * r)

    return Ex


def general_bessel_dE_xdx(x, y, z, E0, wavelength, order):
    kz, kt, r, a, b, phi = general_bessel_constants(wavelength, x, y, kt_by_kz)
    l = order
    dExdx = (
        a
        * E0
        * cmath.exp(1j * kz * z)
        * cmath.exp(1j * l * phi)
        * (
            (1j * l * -y / (x ** 2 + y ** 2)) * jv(l, kt * r)
            + jvp(l, kt * r, 1) * kt * x * (x ** 2 + y ** 2) ** (-1 / 2)
        )
    )

    return dExdx


def general_bessel_dE_xdy(x, y, z, E0, wavelength, order):
    kz, kt, r, a, b, phi = general_bessel_constants(wavelength, x, y, kt_by_kz)
    l = order
    dExdy = (
        a
        * E0
        * cmath.exp(1j * kz * z)
        * cmath.exp(1j * l * phi)
        * (
            (1j * l * x / (x ** 2 + y ** 2)) * jv(l, kt * r)
            + jvp(l, kt * r, 1) * kt * y * (x ** 2 + y ** 2) ** (-1 / 2)
        )
    )

    return dExdy


def general_bessel_dE_xdz(x, y, z, E0, wavelength, order):
    kz, kt, r, a, b, phi = general_bessel_constants(wavelength, x, y, kt_by_kz)
    l = order
    dExdz = (
        E0
        * a
        * cmath.exp(1j * l * phi)
        * jv(l, kt * r)
        * 1j
        * kz
        * cmath.exp(1j * kz * z)
    )

    return dExdz


def general_bessel_E_y(x, y, z, E0, wavelength, order):
    kz, kt, r, a, b, phi = general_bessel_constants(wavelength, x, y, kt_by_kz)
    l = order
    Ey = E0 * cmath.exp(1j * kz * z) * cmath.exp(1j * l * phi) * b * jv(l, kt * r)

    return Ey


def general_bessel_dE_ydx(x, y, z, E0, wavelength, order):
    kz, kt, r, a, b, phi = general_bessel_constants(wavelength, x, y, kt_by_kz)
    l = order
    dExdx = (
        b
        * E0
        * cmath.exp(1j * kz * z)
        * cmath.exp(1j * l * phi)
        * (
            (1j * l * -y / (x ** 2 + y ** 2)) * jv(l, kt * r)
            + jvp(l, kt * r, 1) * kt * x * (x ** 2 + y ** 2) ** (-1 / 2)
        )
    )

    return dExdx


def general_bessel_dE_ydy(x, y, z, E0, wavelength, order):
    kz, kt, r, a, b, phi = general_bessel_constants(wavelength, x, y, kt_by_kz)
    l = order
    dExdy = (
        b
        * E0
        * cmath.exp(1j * kz * z)
        * cmath.exp(1j * l * phi)
        * (
            (1j * l * x / (x ** 2 + y ** 2)) * jv(l, kt * r)
            + jvp(l, kt * r, 1) * kt * y * (x ** 2 + y ** 2) ** (-1 / 2)
        )
    )

    return dExdy


def general_bessel_dE_ydz(x, y, z, E0, wavelength, order):
    kz, kt, r, a, b, phi = general_bessel_constants(wavelength, x, y, kt_by_kz)
    l = order
    dExdz = (
        E0
        * b
        * cmath.exp(1j * l * phi)
        * jv(l, kt * r)
        * 1j
        * kz
        * cmath.exp(1j * kz * z)
    )

    return dExdz


def general_bessel_E_z(x, y, z, E0, wavelength, order):
    kz, kt, r, a, b, phi = general_bessel_constants(wavelength, x, y, kt_by_kz)
    l = order
    Ez = (
        E0
        * (kt / kz)
        * cmath.exp(1j * kz * z)
        * 0.5j
        * cmath.exp(1j * l * phi)
        * (
            (a + 1j * b) * cmath.exp(-1j * phi) * jv(l - 1, kt * r)
            - (a - 1j * b) * cmath.exp(1j * phi) * jv(l + 1, kt * r)
        )
    )

    return Ez


def general_bessel_dE_zdx(x, y, z, E0, wavelength, order):
    kz, kt, r, a, b, phi = general_bessel_constants(wavelength, x, y, kt_by_kz)
    l = order
    dEzdx = (
        E0
        * cmath.exp(1j * kz * z)
        * 0.5j
        * (kt / kz)
        * (
            cmath.exp(1j * l * phi)
            * (
                (a + 1j * b)
                * (
                    -1j
                    * cmath.exp(-1j * phi)
                    * -y
                    / (x ** 2 + y ** 2)
                    * jv(l - 1, kt * r)
                    + cmath.exp(-1j * phi)
                    * jvp(l - 1, kt * r, 1)
                    * kt
                    * x
                    * (x ** 2 + y ** 2) ** (-1 / 2)
                )
                - (a - 1j * b)
                * (
                    1j
                    * cmath.exp(1j * phi)
                    * -y
                    / (x ** 2 + y ** 2)
                    * jv(l + 1, kt * r)
                    + cmath.exp(1j * phi)
                    * jvp(l + 1, kt * r, 1)
                    * kt
                    * x
                    * (x ** 2 + y ** 2) ** (-1 / 2)
                )
            )
            + 1j
            * l
            * cmath.exp(1j * l * phi)
            * -y
            / (x ** 2 + y ** 2)
            * (
                (a + 1j * b) * cmath.exp(-1j * phi) * jv(l - 1, kt * r)
                - (a - 1j * b) * cmath.exp(1j * phi) * jv(l + 1, kt * r)
            )
        )
    )

    # dEzdx_old = (
    #     E0
    #     * cmath.exp(1j * kz * z)
    #     * (
    #         (1j * l * cmath.exp(1j * l * phi) * -y * x ** 2 / (x ** 4 + y ** 2))
    #         * (
    #             (a + 1j * b) * (x - 1j * y) / r * jv(l - 1, kt * r)
    #             - (a - 1j * b) * (x + 1j * y) / r * jv(l + 1, kt * r)
    #         )
    #         + cmath.exp(1j * l * phi)
    #         * (
    #             (a + 1j * b)
    #             * (
    #                 (
    #                     (x ** 2 + y ** 2) ** (-1 / 2)
    #                     - x * (x - 1j * y) * (x ** 2 + y ** 2) ** (-3 / 2)
    #                 )
    #                 * jv(l - 1, kt * r)
    #                 + (x - 1j * y)
    #                 / r
    #                 * jvp(l - 1, kt * r, 1)
    #                 * kt
    #                 * x
    #                 * (x ** 2 + y ** 2) ** (-1 / 2)
    #             )
    #             - (a - 1j * b)
    #             * (
    #                 (
    #                     (x ** 2 + y ** 2) ** (-1 / 2)
    #                     - x * (x + 1j * y) * (x ** 2 + y ** 2) ** (-3 / 2)
    #                 )
    #                 * jv(l + 1, kt * r)
    #                 + (x + 1j * y)
    #                 / r
    #                 * jvp(l + 1, kt * r, 1)
    #                 * kt
    #                 * x
    #                 * (x ** 2 + y ** 2) ** (-1 / 2)
    #             )
    #         )
    #     )
    #     * 1j
    #     / 2
    #     * (kt / kz)
    # )

    return dEzdx


def general_bessel_dE_zdy(x, y, z, E0, wavelength, order):
    kz, kt, r, a, b, phi = general_bessel_constants(wavelength, x, y, kt_by_kz)
    l = order
    dEzdy = (
        E0
        * cmath.exp(1j * kz * z)
        * 0.5j
        * (kt / kz)
        * (
            cmath.exp(1j * l * phi)
            * (
                (a + 1j * b)
                * (
                    -1j
                    * cmath.exp(-1j * phi)
                    * x
                    / (x ** 2 + y ** 2)
                    * jv(l - 1, kt * r)
                    + cmath.exp(-1j * phi)
                    * jvp(l - 1, kt * r, 1)
                    * kt
                    * y
                    * (x ** 2 + y ** 2) ** (-1 / 2)
                )
                - (a - 1j * b)
                * (
                    1j * cmath.exp(1j * phi) * x / (x ** 2 + y ** 2) * jv(l + 1, kt * r)
                    + cmath.exp(1j * phi)
                    * jvp(l + 1, kt * r, 1)
                    * kt
                    * y
                    * (x ** 2 + y ** 2) ** (-1 / 2)
                )
            )
            + 1j
            * l
            * cmath.exp(1j * l * phi)
            * x
            / (x ** 2 + y ** 2)
            * (
                (a + 1j * b) * cmath.exp(-1j * phi) * jv(l - 1, kt * r)
                - (a - 1j * b) * cmath.exp(1j * phi) * jv(l + 1, kt * r)
            )
        )
    )
    # dEzdy_old = (
    #     E0
    #     * cmath.exp(1j * kz * z)
    #     * (
    #         (1j * l * (cmath.exp(1j * l * phi)) * x / (x ** 2 + y ** 2))
    #         * (
    #             (a + 1j * b) * (x - 1j * y) / r * jv(l - 1, kt * r)
    #             - (a - 1j * b) * (x + 1j * y) / r * jv(l + 1, kt * r)
    #         )
    #         + cmath.exp(1j * l * phi)
    #         * (
    #             (a + 1j * b)
    #             * (
    #                 (
    #                     (x ** 2 + y ** 2) ** (-1 / 2)
    #                     + y * (x - 1j * y) * (x ** 2 + y ** 2) ** (-3 / 2)
    #                 )
    #                 * jv(l - 1, kt * r)
    #                 + (x - 1j * y)
    #                 / r
    #                 * jvp(l - 1, kt * r, 1)
    #                 * kt
    #                 * y
    #                 * (x ** 2 + y ** 2) ** (-1 / 2)
    #             )
    #             - (a - 1j * b)
    #             * (
    #                 (
    #                     (x ** 2 + y ** 2) ** (-1 / 2)
    #                     - y * (x + 1j * y) * (x ** 2 + y ** 2) ** (-3 / 2)
    #                 )
    #                 * jv(l + 1, kt * r)
    #                 + (x + 1j * y)
    #                 / r
    #                 * jvp(l + 1, kt * r, 1)
    #                 * kt
    #                 * y
    #                 * (x ** 2 + y ** 2) ** (-1 / 2)
    #             )
    #         )
    #     )
    #     * 1j
    #     / 2
    #     * (kt / kz)
    # )

    return dEzdy


def general_bessel_dE_zdz(x, y, z, E0, wavelength, order):
    kz, kt, r, a, b, phi = general_bessel_constants(wavelength, x, y, kt_by_kz)
    l = order
    dEzdz = (
        E0
        * (kt / kz)
        * cmath.exp(1j * kz * z)
        * 0.5j
        * cmath.exp(1j * l * phi)
        * (
            (a + 1j * b) * cmath.exp(-1j * phi) * jv(l - 1, kt * r)
            - (a - 1j * b) * cmath.exp(1j * phi) * jv(l + 1, kt * r)
        )
        * 1j
        * kz
    )

    return dEzdz


def optical_force(gradE_transpose, E, alpha):
    """Calulates the optical force from the TRANSPOSE of the gradient of the  field"""
    Force = np.zeros(3)
    p = alpha * E
    Force[0] = (1 / 2) * np.real(
        p[0] * gradE_transpose[0, 0]
        + p[1] * gradE_transpose[0, 1]
        + p[2] * gradE_transpose[0, 2]
    )
    Force[1] = (1 / 2) * np.real(
        p[0] * gradE_transpose[1, 0]
        + p[1] * gradE_transpose[1, 1]
        + p[2] * gradE_transpose[1, 2]
    )
    Force[2] = (1 / 2) * np.real(
        p[0] * gradE_transpose[2, 0]
        + p[1] * gradE_transpose[2, 1]
        + p[2] * gradE_transpose[2, 2]
    )
    return Force


def plot_intensity_xy(nx, ny, num_plots, beam):
    Ex = np.zeros((nx, ny), dtype=complex)
    Ey = np.zeros((nx, ny), dtype=complex)
    Ez = np.zeros((nx, ny), dtype=complex)
    z = np.linspace(2e-6, 2e-6, num_plots)
    I = []
    E = []
    fig, ax = plt.subplots(1, num_plots, subplot_kw={"projection": "3d"})
    for k in range(num_plots):

        if beam == "gaussian":
            x = np.linspace(-20e-5, 20e-5, nx)
            y = np.linspace(-20e-5, 20e-5, ny)
            # z = np.linspace(0,0,nx)
            for l in range(n_beams):
                x_prime, y_prime, z_prime = coord_transformation(
                    (x, y, z), beam_angles[l], beam_positions[l]
                )
                for i in range(nx):
                    for j in range(ny):
                        Ex[i][j] = gaussian_E_x(
                            x_prime[i], y_prime[j], z[k], E0, wavelength, w0
                        )
                        Ey[i][j] = gaussian_E_y(
                            x_prime[i], y_prime[j], z[k], E0, wavelength, w0
                        )
                        Ez[i][j] = gaussian_E_z(
                            x_prime[i], y_prime[j], z[k], E0, wavelength, w0
                        )
                E.append(np.array([Ex, Ey, Ez]))
            E_tot = np.sum(E, axis=0)

        elif beam == "bessel":
            x = np.linspace(-1e-6, 1e-6, nx)
            y = np.linspace(-1e-6, 1e-6, ny)
            for i in range(nx):
                for j in range(ny):
                    Ex[i][j] = bessel_E_x(x[i], y[j], z[k], E0, wavelength, alpha_by_k)
                    Ey[i][j] = bessel_E_y(x[i], y[j], z[k], E0, wavelength, alpha_by_k)
                    Ez[i][j] = bessel_E_z(x[i], y[j], z[k], E0, wavelength, alpha_by_k)

        elif beam == "first bessel":
            x = np.linspace(lower, upper, nx)
            y = np.linspace(lower, upper, ny)
            for i in range(nx):
                for j in range(ny):
                    Ex[i][j] = first_bessel_E_x(x[i], y[j], z[k], E0, wavelength)
                    Ey[i][j] = first_bessel_E_y(x[i], y[j], z[k], E0, wavelength)
                    Ez[i][j] = first_bessel_E_z(x[i], y[j], z[k], E0, wavelength)

        elif beam == "general bessel":
            x = np.linspace(lower, upper, nx)
            y = np.linspace(lower, upper, ny)
            for i in range(nx):
                for j in range(ny):
                    Ex[i][j] = general_bessel_E_x(
                        x[i], y[j], z[k], E0, wavelength, order
                    )
                    Ey[i][j] = general_bessel_E_y(
                        x[i], y[j], z[k], E0, wavelength, order
                    )
                    Ez[i][j] = general_bessel_E_z(
                        x[i], y[j], z[k], E0, wavelength, order
                    )

        elif beam == "volke":
            E = np.zeros((nx, ny))
            x = np.linspace(-2e-5, 2e-5, nx)
            y = np.linspace(-2e-5, 2e-5, ny)
            for i in range(nx):
                for j in range(ny):
                    Ex[i][j], Ey[i][j], Ez[i][j] = volke_bessel(x[i], y[j], z[k], E0)

        X, Y = np.meshgrid(x / wavelength, y / wavelength, indexing="ij")
        # Ex, Ey, Ez = E_tot[0], E_tot[1], E_tot[2]
        I.append(np.square(np.abs(Ex)) + np.square(np.abs(Ey)) + np.square(np.abs(Ez)))

        I0 = np.max(I)
        if num_plots > 1:
            ax[k].plot_surface(
                X, Y, I[k] / I0, cmap=cm.coolwarm, linewidth=0, antialiased=False
            )
            ax[k].set_xlabel("x / wavelength")
            ax[k].set_ylabel("y / wavelength")
            ax[k].set_zlabel("Relative Intensity")
            ax[k].set_zlim(0, 1)
            ax[k].set_title("z = {:.1e}".format(z[k]))

        else:
            ax.plot_surface(
                X, Y, I[k] / I0, cmap=cm.coolwarm, linewidth=0, antialiased=False
            )
            ax.set_xlabel("x / wavelength")
            ax.set_ylabel("y / wavelength")
            ax.set_zlabel("Relative Intensity")
            ax.set_zlim(0, 1)
            ax.set_title("z = {:.1e}".format(z[k]))

    plt.show()


def plot_intensity_xy_contour(nx, ny, num_plots, beam):
    Ex = np.zeros((nx, ny), dtype=complex)
    Ey = np.zeros((nx, ny), dtype=complex)
    Ez = np.zeros((nx, ny), dtype=complex)
    z = np.linspace(2e-6, 2e-6, num_plots)
    I = []
    E = []
    fig, ax = plt.subplots(1, num_plots)
    for k in range(num_plots):

        if beam == "gaussian":
            x = np.linspace(lower, upper, nx)
            y = np.linspace(lower, upper, ny)
            # z = np.linspace(0,0,nx)
            for l in range(n_beams):
                x_prime, y_prime, z_prime = coord_transformation(
                    (x, y, z), beam_angles[l], beam_positions[l]
                )
                for i in range(nx):
                    for j in range(ny):
                        Ex[i][j] = gaussian_E_x(
                            x_prime[i], y_prime[j], z[k], E0, wavelength, w0
                        )
                        Ey[i][j] = gaussian_E_y(
                            x_prime[i], y_prime[j], z[k], E0, wavelength, w0
                        )
                        Ez[i][j] = gaussian_E_z(
                            x_prime[i], y_prime[j], z[k], E0, wavelength, w0
                        )
                E.append(np.array([Ex, Ey, Ez]))
            E_tot = np.sum(E, axis=0)

        elif beam == "bessel":
            x = np.linspace(-1e-6, 1e-6, nx)
            y = np.linspace(-1e-6, 1e-6, ny)
            for i in range(nx):
                for j in range(ny):
                    Ex[i][j] = bessel_E_x(x[i], y[j], z[k], E0, wavelength, alpha_by_k)
                    Ey[i][j] = bessel_E_y(x[i], y[j], z[k], E0, wavelength, alpha_by_k)
                    Ez[i][j] = bessel_E_z(x[i], y[j], z[k], E0, wavelength, alpha_by_k)

        elif beam == "first bessel":
            x = np.linspace(lower, upper, nx)
            y = np.linspace(lower, upper, ny)
            for i in range(nx):
                for j in range(ny):
                    Ex[i][j] = first_bessel_E_x(x[i], y[j], z[k], E0, wavelength)
                    Ey[i][j] = first_bessel_E_y(x[i], y[j], z[k], E0, wavelength)
                    Ez[i][j] = first_bessel_E_z(x[i], y[j], z[k], E0, wavelength)

        elif beam == "general bessel":
            x = np.linspace(lower, upper, nx)
            y = np.linspace(lower, upper, ny)
            for i in range(nx):
                for j in range(ny):
                    Ex[i][j] = general_bessel_E_x(
                        x[i], y[j], z[k], E0, wavelength, order
                    )
                    Ey[i][j] = general_bessel_E_y(
                        x[i], y[j], z[k], E0, wavelength, order
                    )
                    Ez[i][j] = general_bessel_E_z(
                        x[i], y[j], z[k], E0, wavelength, order
                    )

        elif beam == "volke":
            E = np.zeros((nx, ny))
            x = np.linspace(-2e-5, 2e-5, nx)
            y = np.linspace(-2e-5, 2e-5, ny)
            for i in range(nx):
                for j in range(ny):
                    Ex[i][j], Ey[i][j], Ez[i][j] = volke_bessel(x[i], y[j], z[k], E0)

        X, Y = np.meshgrid(x, y, indexing="ij")
        # Ex, Ey, Ez = E_tot[0], E_tot[1], E_tot[2]
        I.append(np.square(np.abs(Ex)) + np.square(np.abs(Ey)) + np.square(np.abs(Ez)))

        I0 = np.max(I)
        if num_plots > 1:
            ax[k].plot_surface(
                X, Y, I[k] / I0, cmap=cm.coolwarm, linewidth=0, antialiased=False
            )
            ax[k].set_xlabel("x / wavelength")
            ax[k].set_ylabel("y / wavelength")
            ax[k].set_zlabel("Relative Intensity")
            ax[k].set_zlim(0, 1)
            ax[k].set_title("z = {:.1e}".format(z[k]))

        else:
#            ax.axis('equal')
            ax.set_aspect('equal','box')
            cs=ax.contourf(X, Y, I[k], cmap=cm.summer, levels=30)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            cbar = fig.colorbar(cs)
            # ax.set_zlabel("Relative Intensity")
            # ax.set_zlim(0, 1)
            # ax.set_title("z = {:.1e}".format(z[k]))
    return ax
    # plt.show()


def intensity_xy_animation(nx, ny, frames, beam):
    x = np.linspace(-20e-6, 20e-6, nx)
    y = np.linspace(-20e-6, 20e-6, ny)
    Ex = np.zeros((nx, ny), dtype=complex)
    Ey = np.zeros((nx, ny), dtype=complex)
    Ez = np.zeros((nx, ny), dtype=complex)
    z = np.linspace(-100e-6, 100e-6, frames)
    I = []

    for k in range(frames):
        if beam == "gaussian":
            for i in range(nx):
                for j in range(ny):
                    Ex[i][j] = gaussian_E_x(x[i], y[j], z[k], E0, wavelength, w0)
                    Ey[i][j] = gaussian_E_y(x[i], y[j], z[k], E0, wavelength, w0)
                    # Ez[i][j] = gaussian_E_z(x[i], y[j], z[k], E0, wavelength, w0)

        elif beam == "bessel":
            for i in range(nx):
                for j in range(ny):
                    Ex[i][j] = bessel_E_x(x[i], y[j], z[k], E0, wavelength)
                    Ey[i][j] = bessel_E_y(x[i], y[j], z[k], E0, wavelength)
                    Ez[i][j] = gaussian_E_z(x[i], y[j], z[k], E0, wavelength, w0)

        X, Y = np.meshgrid(x / wavelength, y / wavelength)
        I.append(np.square(np.abs(Ex)) + np.square(np.abs(Ey)) + np.square(np.abs(Ez)))

        I0 = np.max(I)

    return X, Y, I / I0, z


def update(i):
    """
    Update function tells the animator what to change in every frame.
    """
    ax.clear()
    surf = ax.plot_surface(X, Y, I[i], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_title("z = {:.1e}".format(z[i]))
    ax.set_zlim(0, 1)
    return surf


def plot_intensity_xz(nx, nz, y):

    x = np.linspace(-2e-6, 2e-6, nx)
    z = np.linspace(-4e-6, 4e-6, nz)
    Ex = np.zeros((nx, nz), dtype=complex)
    Ey = np.zeros((nx, nz), dtype=complex)
    Ez = np.zeros((nx, nz), dtype=complex)

    for i in range(nx):
        for j in range(nz):
            Ex[i][j] = gaussian_E_x(x[i], y, z[j], E0, wavelength, w0)
            Ey[i][j] = gaussian_E_y(x[i], y, z[j], E0, wavelength, w0)
            Ez[i][j] = gaussian_E_z(x[i], y, z[j], E0, wavelength, w0)

    X, Z = np.meshgrid(x / wavelength, z / wavelength)
    I = np.square(np.abs(Ex)) + np.square(np.abs(Ey) + np.square(np.abs(Ez)))
    I0 = np.max(I)
    fig, ax = plt.subplots(
        subplot_kw={"projection": "3d"}
    )  # Intensity plot of a cross section of the beam at z
    ax.plot_surface(X, Z, I / I0, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel("x/wavelength")
    ax.set_ylabel("z/wavelength")
    plt.show()


def force_z_curve(x_position, y_position, z_start, z_end, num, beam):
    z_position = np.linspace(z_start, z_end, num)
    Fx_distance_array = []
    Fy_distance_array = []
    Fz_distance_array = []
    for i in range(num):
        if beam == "gaussian":
            Ex = gaussian_E_x(x_position, y_position, z_position[i], E0, wavelength, w0)
            Ey = gaussian_E_y(x_position, y_position, z_position[i], E0, wavelength, w0)
            Ez = gaussian_E_z(x_position, y_position, z_position[i], E0, wavelength, w0)

            dExdx = gaussian_dE_xdx(
                x_position, y_position, z_position[i], E0, wavelength, w0
            )
            dEydx = gaussian_dE_ydx(
                x_position, y_position, z_position[i], E0, wavelength, w0
            )
            dExdy = gaussian_dE_xdy(
                x_position, y_position, z_position[i], E0, wavelength, w0
            )
            dEydy = gaussian_dE_ydy(
                x_position, y_position, z_position[i], E0, wavelength, w0
            )
            dExdz = gaussian_dE_xdz(
                x_position, y_position, z_position[i], E0, wavelength, w0
            )
            dEydz = gaussian_dE_ydz(
                x_position, y_position, z_position[i], E0, wavelength, w0
            )
            dEzdx = gaussian_dE_zdx(
                x_position, y_position, z_position[i], E0, wavelength, w0
            )
            dEzdy = gaussian_dE_zdy(
                x_position, y_position, z_position[i], E0, wavelength, w0
            )
            dEzdz = gaussian_dE_zdz(
                x_position, y_position, z_position[i], E0, wavelength, w0
            )

        if beam == "bessel":
            Ex = bessel_E_x(
                x_position, y_position, z_position[i], E0, wavelength, alpha_by_k
            )
            Ey = bessel_E_y(
                x_position, y_position, z_position[i], E0, wavelength, alpha_by_k
            )
            Ez = bessel_E_z(
                x_position, y_position, z_position[i], E0, wavelength, alpha_by_k
            )
            dExdx = bessel_dE_xdx(
                x_position, y_position, z_position[i], E0, wavelength, alpha_by_k
            )
            dEydx = bessel_dE_ydx(
                x_position, y_position, z_position[i], E0, wavelength, alpha_by_k
            )
            dExdy = bessel_dE_xdy(
                x_position, y_position, z_position[i], E0, wavelength, alpha_by_k
            )
            dEydy = bessel_dE_ydy(
                x_position, y_position, z_position[i], E0, wavelength, alpha_by_k
            )
            dExdz = bessel_dE_xdz(
                x_position, y_position, z_position[i], E0, wavelength, alpha_by_k
            )
            dEydz = bessel_dE_ydz(
                x_position, y_position, z_position[i], E0, wavelength, alpha_by_k
            )
            dEzdx = bessel_dE_zdx(
                x_position, y_position, z_position[i], E0, wavelength, alpha_by_k
            )
            dEzdy = bessel_dE_zdy(
                x_position, y_position, z_position[i], E0, wavelength, alpha_by_k
            )
            dEzdz = bessel_dE_zdz(
                x_position, y_position, z_position[i], E0, wavelength, alpha_by_k
            )
            # except ZeroDivisionError:
            #     i = i+1
            #     continue

        if beam == "first bessel":
            Ex = first_bessel_E_x(x_position, y_position, z_position[i], E0, wavelength)
            Ey = first_bessel_E_y(x_position, y_position, z_position[i], E0, wavelength)
            Ez = first_bessel_E_z(x_position, y_position, z_position[i], E0, wavelength)

            dExdx = first_bessel_dE_xdx(
                x_position, y_position, z_position[i], E0, wavelength
            )
            dEydx = first_bessel_dE_ydx(
                x_position, y_position, z_position[i], E0, wavelength
            )
            dExdy = first_bessel_dE_xdy(
                x_position, y_position, z_position[i], E0, wavelength
            )
            dEydy = first_bessel_dE_ydy(
                x_position, y_position, z_position[i], E0, wavelength
            )
            dExdz = first_bessel_dE_xdz(
                x_position, y_position, z_position[i], E0, wavelength
            )
            dEydz = first_bessel_dE_ydz(
                x_position, y_position, z_position[i], E0, wavelength
            )
            dEzdx = first_bessel_dE_zdx(
                x_position, y_position, z_position[i], E0, wavelength
            )
            dEzdy = first_bessel_dE_zdy(
                x_position, y_position, z_position[i], E0, wavelength
            )
            dEzdz = first_bessel_dE_zdz(
                x_position, y_position, z_position[i], E0, wavelength
            )

        if beam == "general bessel":
            Ex = general_bessel_E_x(
                x_position, y_position, z_position[i], E0, wavelength, order
            )
            Ey = general_bessel_E_y(
                x_position, y_position, z_position[i], E0, wavelength, order
            )
            Ez = general_bessel_E_z(
                x_position, y_position, z_position[i], E0, wavelength, order
            )

            dExdx = general_bessel_dE_xdx(
                x_position, y_position, z_position[i], E0, wavelength, order
            )
            dEydx = general_bessel_dE_ydx(
                x_position, y_position, z_position[i], E0, wavelength, order
            )
            dExdy = general_bessel_dE_xdy(
                x_position, y_position, z_position[i], E0, wavelength, order
            )
            dEydy = general_bessel_dE_ydy(
                x_position, y_position, z_position[i], E0, wavelength, order
            )
            dExdz = general_bessel_dE_xdz(
                x_position, y_position, z_position[i], E0, wavelength, order
            )
            dEydz = general_bessel_dE_ydz(
                x_position, y_position, z_position[i], E0, wavelength, order
            )
            dEzdx = general_bessel_dE_zdx(
                x_position, y_position, z_position[i], E0, wavelength, order
            )
            dEzdy = general_bessel_dE_zdy(
                x_position, y_position, z_position[i], E0, wavelength, order
            )
            dEzdz = general_bessel_dE_zdz(
                x_position, y_position, z_position[i], E0, wavelength, order
            )
        gradE = np.zeros((3, 3), dtype=complex)
        gradE[0][0] = np.conjugate(dExdx)
        gradE[0][1] = np.conjugate(dExdy)
        gradE[0][2] = np.conjugate(dExdz)
        gradE[1][0] = np.conjugate(dEydx)
        gradE[1][1] = np.conjugate(dEydy)
        gradE[1][2] = np.conjugate(dEydz)
        gradE[2][0] = np.conjugate(dEzdx)
        gradE[2][1] = np.conjugate(dEzdy)
        gradE[2][2] = np.conjugate(dEzdz)

        E = np.array(([Ex], [Ey], [Ez]))
        force_on_particle = optical_force(gradE.T, E, a)
        Fx = force_on_particle[0]
        Fy = force_on_particle[1]
        Fz = force_on_particle[2]
        Fx_distance_array.append(Fx)
        Fy_distance_array.append(Fy)
        Fz_distance_array.append(Fz)

    return z_position, Fx_distance_array, Fy_distance_array, Fz_distance_array


def force_y_curve(x_position, z_position, y_start, y_end, num, beam):
    y_position = np.linspace(y_start, y_end, num)
    Fx_distance_array = []
    Fy_distance_array = []
    Fz_distance_array = []
    for i in range(num):
        if beam == "gaussian":
            Ex = gaussian_E_x(x_position, y_position[i], z_position, E0, wavelength, w0)
            Ey = gaussian_E_y(x_position, y_position[i], z_position, E0, wavelength, w0)
            Ez = gaussian_E_z(x_position, y_position[i], z_position, E0, wavelength, w0)

            dExdx = gaussian_dE_xdx(
                x_position, y_position[i], z_position, E0, wavelength, w0
            )
            dEydx = gaussian_dE_ydx(
                x_position, y_position[i], z_position, E0, wavelength, w0
            )
            dExdy = gaussian_dE_xdy(
                x_position, y_position[i], z_position, E0, wavelength, w0
            )
            dEydy = gaussian_dE_ydy(
                x_position, y_position[i], z_position, E0, wavelength, w0
            )
            dExdz = gaussian_dE_xdz(
                x_position, y_position[i], z_position, E0, wavelength, w0
            )
            dEydz = gaussian_dE_ydz(
                x_position, y_position[i], z_position, E0, wavelength, w0
            )
            dEzdx = gaussian_dE_zdx(
                x_position, y_position[i], z_position, E0, wavelength, w0
            )
            dEzdy = gaussian_dE_zdy(
                x_position, y_position[i], z_position, E0, wavelength, w0
            )
            dEzdz = gaussian_dE_zdz(
                x_position, y_position[i], z_position, E0, wavelength, w0
            )

        if beam == "bessel":
            Ex = bessel_E_x(
                x_position, y_position[i], z_position, E0, wavelength, alpha_by_k
            )
            Ey = bessel_E_y(
                x_position, y_position[i], z_position, E0, wavelength, alpha_by_k
            )
            Ez = bessel_E_z(
                x_position, y_position[i], z_position, E0, wavelength, alpha_by_k
            )

            dExdx = bessel_dE_xdx(
                x_position, y_position[i], z_position, E0, wavelength, alpha_by_k
            )
            dEydx = bessel_dE_ydx(
                x_position, y_position[i], z_position, E0, wavelength, alpha_by_k
            )
            dExdy = bessel_dE_xdy(
                x_position, y_position[i], z_position, E0, wavelength, alpha_by_k
            )
            dEydy = bessel_dE_ydy(
                x_position, y_position[i], z_position, E0, wavelength, alpha_by_k
            )
            dExdz = bessel_dE_xdz(
                x_position, y_position[i], z_position, E0, wavelength, alpha_by_k
            )
            dEydz = bessel_dE_ydz(
                x_position, y_position[i], z_position, E0, wavelength, alpha_by_k
            )
            dEzdx = bessel_dE_zdx(
                x_position, y_position[i], z_position, E0, wavelength, alpha_by_k
            )
            dEzdy = bessel_dE_zdy(
                x_position, y_position[i], z_position, E0, wavelength, alpha_by_k
            )
            dEzdz = bessel_dE_zdz(
                x_position, y_position[i], z_position, E0, wavelength, alpha_by_k
            )

        if beam == "first bessel":
            Ex = first_bessel_E_x(x_position, y_position[i], z_position, E0, wavelength)
            Ey = first_bessel_E_y(x_position, y_position[i], z_position, E0, wavelength)
            Ez = first_bessel_E_z(x_position, y_position[i], z_position, E0, wavelength)

            dExdx = first_bessel_dE_xdx(
                x_position, y_position[i], z_position, E0, wavelength
            )
            dEydx = first_bessel_dE_ydx(
                x_position, y_position[i], z_position, E0, wavelength
            )
            dExdy = first_bessel_dE_xdy(
                x_position, y_position[i], z_position, E0, wavelength
            )
            dEydy = first_bessel_dE_ydy(
                x_position, y_position[i], z_position, E0, wavelength
            )
            dExdz = first_bessel_dE_xdz(
                x_position, y_position[i], z_position, E0, wavelength
            )
            dEydz = first_bessel_dE_ydz(
                x_position, y_position[i], z_position, E0, wavelength
            )
            dEzdx = first_bessel_dE_zdx(
                x_position, y_position[i], z_position, E0, wavelength
            )
            dEzdy = first_bessel_dE_zdy(
                x_position, y_position[i], z_position, E0, wavelength
            )
            dEzdz = first_bessel_dE_zdz(
                x_position, y_position[i], z_position, E0, wavelength
            )

        if beam == "general bessel":
            Ex = general_bessel_E_x(
                x_position, y_position[i], z_position, E0, wavelength, order
            )
            Ey = general_bessel_E_y(
                x_position, y_position[i], z_position, E0, wavelength, order
            )
            Ez = general_bessel_E_z(
                x_position, y_position[i], z_position, E0, wavelength, order
            )

            dExdx = general_bessel_dE_xdx(
                x_position, y_position[i], z_position, E0, wavelength, order
            )
            dEydx = general_bessel_dE_ydx(
                x_position, y_position[i], z_position, E0, wavelength, order
            )
            dExdy = general_bessel_dE_xdy(
                x_position, y_position[i], z_position, E0, wavelength, order
            )
            dEydy = general_bessel_dE_ydy(
                x_position, y_position[i], z_position, E0, wavelength, order
            )
            dExdz = general_bessel_dE_xdz(
                x_position, y_position[i], z_position, E0, wavelength, order
            )
            dEydz = general_bessel_dE_ydz(
                x_position, y_position[i], z_position, E0, wavelength, order
            )
            dEzdx = general_bessel_dE_zdx(
                x_position, y_position[i], z_position, E0, wavelength, order
            )
            dEzdy = general_bessel_dE_zdy(
                x_position, y_position[i], z_position, E0, wavelength, order
            )
            dEzdz = general_bessel_dE_zdz(
                x_position, y_position[i], z_position, E0, wavelength, order
            )
        gradE = np.zeros((3, 3), dtype=complex)
        gradE[0][0] = np.conjugate(dExdx)
        gradE[0][1] = np.conjugate(dExdy)
        gradE[0][2] = np.conjugate(dExdz)
        gradE[1][0] = np.conjugate(dEydx)
        gradE[1][1] = np.conjugate(dEydy)
        gradE[1][2] = np.conjugate(dEydz)
        gradE[2][0] = np.conjugate(dEzdx)
        gradE[2][1] = np.conjugate(dEzdy)
        gradE[2][2] = np.conjugate(dEzdz)

        E = np.array(([Ex], [Ey], [Ez]), dtype=complex)
        force_on_particle = optical_force(gradE.T, E, a)
        Fx = force_on_particle[0] * 100
        Fy = force_on_particle[1]
        Fz = force_on_particle[2]
        Fx_distance_array.append(Fx)
        Fy_distance_array.append(Fy)
        Fz_distance_array.append(Fz)

    return y_position, Fx_distance_array, Fy_distance_array, Fz_distance_array


def force_x_curve(y_position, z_position, x_start, x_end, num, beam):
    """scans along x and calulates forces in x,y,z directions"""
    x_position = np.linspace(x_start, x_end, num)
    Fx_distance_array = []
    Fy_distance_array = []
    Fz_distance_array = []
    for i in range(num):
        if beam == "gaussian":
            Ex = gaussian_E_x(x_position[i], y_position, z_position, E0, wavelength, w0)
            Ey = gaussian_E_y(x_position[i], y_position, z_position, E0, wavelength, w0)
            Ez = gaussian_E_z(x_position[i], y_position, z_position, E0, wavelength, w0)

            dExdx = gaussian_dE_xdx(
                x_position[i], y_position, z_position, E0, wavelength, w0
            )
            dEydx = gaussian_dE_ydx(
                x_position[i], y_position, z_position, E0, wavelength, w0
            )
            dExdy = gaussian_dE_xdy(
                x_position[i], y_position, z_position, E0, wavelength, w0
            )
            dEydy = gaussian_dE_ydy(
                x_position[i], y_position, z_position, E0, wavelength, w0
            )
            dExdz = gaussian_dE_xdz(
                x_position[i], y_position, z_position, E0, wavelength, w0
            )
            dEydz = gaussian_dE_ydz(
                x_position[i], y_position, z_position, E0, wavelength, w0
            )
            dEzdx = gaussian_dE_zdx(
                x_position[i], y_position, z_position, E0, wavelength, w0
            )
            dEzdy = gaussian_dE_zdy(
                x_position[i], y_position, z_position, E0, wavelength, w0
            )
            dEzdz = gaussian_dE_zdz(
                x_position[i], y_position, z_position, E0, wavelength, w0
            )

        if beam == "bessel":
            Ex = bessel_E_x(
                x_position[i], y_position, z_position, E0, wavelength, alpha_by_k
            )
            Ey = bessel_E_y(
                x_position[i], y_position, z_position, E0, wavelength, alpha_by_k
            )
            Ez = bessel_E_z(
                x_position[i], y_position, z_position, E0, wavelength, alpha_by_k
            )

            dExdx = bessel_dE_xdx(
                x_position[i], y_position, z_position, E0, wavelength, alpha_by_k
            )
            dEydx = bessel_dE_ydx(
                x_position[i], y_position, z_position, E0, wavelength, alpha_by_k
            )
            dExdy = bessel_dE_xdy(
                x_position[i], y_position, z_position, E0, wavelength, alpha_by_k
            )
            dEydy = bessel_dE_ydy(
                x_position[i], y_position, z_position, E0, wavelength, alpha_by_k
            )
            dExdz = bessel_dE_xdz(
                x_position[i], y_position, z_position, E0, wavelength, alpha_by_k
            )
            dEydz = bessel_dE_ydz(
                x_position[i], y_position, z_position, E0, wavelength, alpha_by_k
            )
            dEzdx = bessel_dE_zdx(
                x_position[i], y_position, z_position, E0, wavelength, alpha_by_k
            )
            dEzdy = bessel_dE_zdy(
                x_position[i], y_position, z_position, E0, wavelength, alpha_by_k
            )
            dEzdz = bessel_dE_zdz(
                x_position[i], y_position, z_position, E0, wavelength, alpha_by_k
            )

        if beam == "first bessel":
            Ex = first_bessel_E_x(x_position[i], y_position, z_position, E0, wavelength)
            Ey = first_bessel_E_y(x_position[i], y_position, z_position, E0, wavelength)
            Ez = first_bessel_E_z(x_position[i], y_position, z_position, E0, wavelength)

            dExdx = first_bessel_dE_xdx(
                x_position[i], y_position, z_position, E0, wavelength
            )
            dEydx = first_bessel_dE_ydx(
                x_position[i], y_position, z_position, E0, wavelength
            )
            dExdy = first_bessel_dE_xdy(
                x_position[i], y_position, z_position, E0, wavelength
            )
            dEydy = first_bessel_dE_ydy(
                x_position[i], y_position, z_position, E0, wavelength
            )
            dExdz = first_bessel_dE_xdz(
                x_position[i], y_position, z_position, E0, wavelength
            )
            dEydz = first_bessel_dE_ydz(
                x_position[i], y_position, z_position, E0, wavelength
            )
            dEzdx = first_bessel_dE_zdx(
                x_position[i], y_position, z_position, E0, wavelength
            )
            dEzdy = first_bessel_dE_zdy(
                x_position[i], y_position, z_position, E0, wavelength
            )
            dEzdz = first_bessel_dE_zdz(
                x_position[i], y_position, z_position, E0, wavelength
            )

        if beam == "general bessel":
            Ex = general_bessel_E_x(
                x_position[i], y_position, z_position, E0, wavelength, order
            )
            Ey = general_bessel_E_y(
                x_position[i], y_position, z_position, E0, wavelength, order
            )
            Ez = general_bessel_E_z(
                x_position[i], y_position, z_position, E0, wavelength, order
            )

            dExdx = general_bessel_dE_xdx(
                x_position[i], y_position, z_position, E0, wavelength, order
            )
            dEydx = general_bessel_dE_ydx(
                x_position[i], y_position, z_position, E0, wavelength, order
            )
            dExdy = general_bessel_dE_xdy(
                x_position[i], y_position, z_position, E0, wavelength, order
            )
            dEydy = general_bessel_dE_ydy(
                x_position[i], y_position, z_position, E0, wavelength, order
            )
            dExdz = general_bessel_dE_xdz(
                x_position[i], y_position, z_position, E0, wavelength, order
            )
            dEydz = general_bessel_dE_ydz(
                x_position[i], y_position, z_position, E0, wavelength, order
            )
            dEzdx = general_bessel_dE_zdx(
                x_position[i], y_position, z_position, E0, wavelength, order
            )
            dEzdy = general_bessel_dE_zdy(
                x_position[i], y_position, z_position, E0, wavelength, order
            )
            dEzdz = general_bessel_dE_zdz(
                x_position[i], y_position, z_position, E0, wavelength, order
            )

        gradE = np.zeros((3, 3), dtype=complex)
        gradE[0][0] = np.conjugate(dExdx)
        gradE[0][1] = np.conjugate(dExdy)
        gradE[0][2] = np.conjugate(dExdz)
        gradE[1][0] = np.conjugate(dEydx)
        gradE[1][1] = np.conjugate(dEydy)
        gradE[1][2] = np.conjugate(dEydz)
        gradE[2][0] = np.conjugate(dEzdx)
        gradE[2][1] = np.conjugate(dEzdy)
        gradE[2][2] = np.conjugate(dEzdz)

        E = np.array(([Ex], [Ey], [Ez]))
        force_on_particle = optical_force(gradE.T, E, a)
        Fx = force_on_particle[0]
        Fy = force_on_particle[1]
        Fz = force_on_particle[2]
        Fx_distance_array.append(Fx)
        Fy_distance_array.append(Fy)
        Fz_distance_array.append(Fz)

    return x_position, Fx_distance_array, Fy_distance_array, Fz_distance_array


def force_distance_plots(beam):
    # order 100 for bessel beam, order 1e-5 for gaussian
    x, fxx, fxy, fxz = force_x_curve(0, 0, -2e-6, 2e-6, 200, beam)
    y, fyx, fyy, fyz = force_y_curve(0, 0, -2e-6, 2e-6, 200, beam)
    z, fzx, fzy, fzz = force_z_curve(1e-50, 0, -400, 400, 250, beam)

    fig, ax = plt.subplots(1, 3)
    ax[0].set_xlabel("Distance in x direction")
    ax[1].set_xlabel("Distance in y direction")
    ax[2].set_xlabel("Distance in z direction")
    ax[0].set_ylabel("Force in x direction")
    ax[1].set_ylabel("Force in y direction")
    ax[2].set_ylabel("Force in z direction")
    ax[0].plot(x, fxx)
    ax[1].plot(y, fyy)
    ax[2].plot(z, fzz)
    for label in ax[0].get_xaxis().get_ticklabels()[::2]:
        label.set_visible(False)
    for label in ax[1].get_xaxis().get_ticklabels()[::2]:
        label.set_visible(False)
    for label in ax[2].get_xaxis().get_ticklabels()[::2]:
        label.set_visible(False)

    # fig, ax = plt.subplots(1, 2)
    # ax[0].set_xlabel("Distance in x direction")
    # ax[1].set_xlabel("Distance in x direction")
    # ax[0].set_ylabel("Force in x direction")
    # ax[1].set_ylabel("Force in y direction")
    # ax[0].plot(x, fxx)
    # ax[1].plot(x, fxy)
    # for label in ax[0].get_xaxis().get_ticklabels()[::2]:
    #     label.set_visible(False)
    # for label in ax[1].get_xaxis().get_ticklabels()[::2]:
    #     label.set_visible(False)
    # plt.tight_layout()

    # plt.plot(y, fyy)
    # plt.plot(y, fyx)

    plt.show()


def incident_field(beam, position_array):
    """returns Ex, Ey, Ez field components for the position given (must be a single coordinate). Beam defines which beam shape is used. n_beams allows for multiple beams"""
    E = []
    x_position = position_array[0]
    y_position = position_array[1]
    z_position = position_array[2]
    for i in range(n_beams):
        x, y, z = coord_transformation(
            (x_position, y_position, z_position), beam_angles[i], beam_positions[i]
        )

        if beam == "gaussian":
            # for i in range(len(x_array)):
            Ex = gaussian_E_x(x, y, z, E0, wavelength, w0)
            Ey = gaussian_E_y(x, y, z, E0, wavelength, w0)
            Ez = gaussian_E_z(x, y, z, E0, wavelength, w0)

        if beam == "zeroth bessel":
            Ex = bessel_E_x(x, y, z, E0, wavelength, alpha_by_k)
            Ey = bessel_E_y(x, y, z, E0, wavelength, alpha_by_k)
            Ez = bessel_E_z(x, y, z, E0, wavelength, alpha_by_k)

        if beam == "first bessel":
            Ex = first_bessel_E_x(x, y, z, E0, wavelength)
            Ey = first_bessel_E_y(x, y, z, E0, wavelength)
            Ez = first_bessel_E_z(x, y, z, E0, wavelength)

        if beam == "general bessel":
            Ex = general_bessel_E_x(
                x_position, y_position, z_position, E0, wavelength, order
            )
            Ey = general_bessel_E_y(
                x_position, y_position, z_position, E0, wavelength, order
            )
            Ez = general_bessel_E_z(
                x_position, y_position, z_position, E0, wavelength, order
            )

        E.append(np.array([Ex, Ey, Ez]))

    E_tot = np.sum(E, axis=0)
    return E_tot


def incident_field_gradient(beam, position_array):
    """calculates the gradient of a shaped beam at a position given"""
    x_position = position_array[0]
    y_position = position_array[1]
    z_position = position_array[2]
    gradE_tot = []
    for i in range(n_beams):
        x, y, z = coord_transformation(
            (x_position, y_position, z_position), beam_angles[i], beam_positions[i]
        )
        if beam == "gaussian":
            dExdx = gaussian_dE_xdx(x, y, z, E0, wavelength, w0)
            dEydx = gaussian_dE_ydx(x, y, z, E0, wavelength, w0)
            dExdy = gaussian_dE_xdy(x, y, z, E0, wavelength, w0)
            dEydy = gaussian_dE_ydy(x, y, z, E0, wavelength, w0)
            dExdz = gaussian_dE_xdz(x, y, z, E0, wavelength, w0)
            dEydz = gaussian_dE_ydz(x, y, z, E0, wavelength, w0)
            dEzdx = gaussian_dE_zdx(x, y, z, E0, wavelength, w0)
            dEzdy = gaussian_dE_zdy(x, y, z, E0, wavelength, w0)
            dEzdz = gaussian_dE_zdz(x, y, z, E0, wavelength, w0)

        if beam == "zeroth bessel":
            dExdx = bessel_dE_xdx(x, y, z, E0, wavelength, alpha_by_k)
            dEydx = bessel_dE_ydx(x, y, z, E0, wavelength, alpha_by_k)
            dExdy = bessel_dE_xdy(x, y, z, E0, wavelength, alpha_by_k)
            dEydy = bessel_dE_ydy(x, y, z, E0, wavelength, alpha_by_k)
            dExdz = bessel_dE_xdz(x, y, z, E0, wavelength, alpha_by_k)
            dEydz = bessel_dE_ydz(x, y, z, E0, wavelength, alpha_by_k)
            dEzdx = bessel_dE_zdx(x, y, z, E0, wavelength, alpha_by_k)
            dEzdy = bessel_dE_zdy(x, y, z, E0, wavelength, alpha_by_k)
            dEzdz = bessel_dE_zdz(x, y, z, E0, wavelength, alpha_by_k)

        if beam == "first bessel":
            dExdx = first_bessel_dE_xdx(x, y, z, E0, wavelength)
            dEydx = first_bessel_dE_ydx(x, y, z, E0, wavelength)
            dExdy = first_bessel_dE_xdy(x, y, z, E0, wavelength)
            dEydy = first_bessel_dE_ydy(x, y, z, E0, wavelength)
            dExdz = first_bessel_dE_xdz(x, y, z, E0, wavelength)
            dEydz = first_bessel_dE_ydz(x, y, z, E0, wavelength)
            dEzdx = first_bessel_dE_zdx(x, y, z, E0, wavelength)
            dEzdy = first_bessel_dE_zdy(x, y, z, E0, wavelength)
            dEzdz = first_bessel_dE_zdz(x, y, z, E0, wavelength)

        if beam == "general bessel":
            dExdx = general_bessel_dE_xdx(
                x_position, y_position, z_position, E0, wavelength, order
            )
            dEydx = general_bessel_dE_ydx(
                x_position, y_position, z_position, E0, wavelength, order
            )
            dExdy = general_bessel_dE_xdy(
                x_position, y_position, z_position, E0, wavelength, order
            )
            dEydy = general_bessel_dE_ydy(
                x_position, y_position, z_position, E0, wavelength, order
            )
            dExdz = general_bessel_dE_xdz(
                x_position, y_position, z_position, E0, wavelength, order
            )
            dEydz = general_bessel_dE_ydz(
                x_position, y_position, z_position, E0, wavelength, order
            )
            dEzdx = general_bessel_dE_zdx(
                x_position, y_position, z_position, E0, wavelength, order
            )
            dEzdy = general_bessel_dE_zdy(
                x_position, y_position, z_position, E0, wavelength, order
            )
            dEzdz = general_bessel_dE_zdz(
                x_position, y_position, z_position, E0, wavelength, order
            )

        gradE = np.zeros((3, 3), dtype=complex)
        gradE[0][0] = np.conjugate(dExdx)
        gradE[0][1] = np.conjugate(dExdy)
        gradE[0][2] = np.conjugate(dExdz)
        gradE[1][0] = np.conjugate(dEydx)
        gradE[1][1] = np.conjugate(dEydy)
        gradE[1][2] = np.conjugate(dEydz)
        gradE[2][0] = np.conjugate(dEzdx)
        gradE[2][1] = np.conjugate(dEzdy)
        gradE[2][2] = np.conjugate(dEzdz)

        gradE_tot.append(gradE)

    gradE_sum = np.sum(gradE_tot, axis=0)

    return gradE_sum


def force_tranform(Fx, Fy, x, y):
    """Transforms Fx and Fy components to Fphi component, and expresses this in terms of x and y unit vectors for plotting"""
    phi = arctan2(y, x)
    fx = -Fx * sin(phi)
    fy = Fy * cos(phi)
    F_phi = -Fx * sin(phi) + Fy * cos(phi)
    Fx_prime = F_phi * -sin(phi)
    Fy_prime = F_phi * cos(phi)
    return Fx_prime, Fy_prime


def force_map(beam):
    nx = 50
    ny = 50
    x = np.linspace(lower, upper, nx)
    y = np.linspace(lower, upper, ny)
    z = 0.0#4e-3
    Fx = np.zeros((nx, ny))
    Fy = np.zeros((nx, ny))
    Fx_azimuthal = np.zeros((nx, ny))
    Fy_azimuthal = np.zeros((nx, ny))

    for i in range(nx):
        for j in range(ny):
            gradE = incident_field_gradient(beam, (x[i], y[j], z))
            E = incident_field(beam, (x[i], y[j], z))
            Force = optical_force(gradE.T, E, a)
            #print(x[i],y[j],Force)
            Fx[i][j] = Force[0]  # [j][i] due to how plt.quiver plots the coordinates
            Fy[i][j] = Force[1]
            Fx_temp, Fy_temp = force_tranform(Force[0], Force[1], x[i], y[j])
            Fx_azimuthal[i][j] = Fx_temp
            Fy_azimuthal[i][j] = Fy_temp
#            print(x[i],y[j],Fx_temp,Fy_temp)

    X, Y = np.meshgrid(x, y, indexing="ij")
    ax = plot_intensity_xy_contour(100, 100, 1, beam)
#    ax = plot_intensity_xy_contour(100, 100, 1, beam)
    ax.quiver(X, Y, Fx_azimuthal, Fy_azimuthal, units="x")
#    ax.quiver(X, Y, Fx, Fy, units="x")
    ax.set_title("CP Bessel Beam Order {:d}".format(order))
    # plt.xlabel("x")
    # plt.ylabel("y")
    plt.show()
    return


def coord_transformation(coords, angles, trans):
    """a is angle around x axis, b is angle around y axis, g is angle around z axis"""
    a, b, g = angles[0], angles[1], angles[2]
    dx, dy, dz = trans[0], trans[1], trans[2]

    R_z = np.array(((cos(g), -sin(g), 0), (sin(g), cos(g), 0), (0, 0, 1)))
    R_y = np.array(((cos(b), 0, sin(b)), (0, 1, 0), (-sin(b), 0, cos(b))))
    R_x = np.array(((1, 0, 0), (0, cos(a), -sin(a)), (0, sin(a), cos(a))))

    rotation = R_z @ R_y @ R_x
    rotated_coords = rotation @ np.array(coords, dtype=object)

    x_prime, y_prime, z_prime = rotated_coords[0], rotated_coords[1], rotated_coords[2]

    x_prime = x_prime + dx
    y_prime = y_prime + dy
    z_prime = z_prime + dz

    # x_prime, y_prime, z_prime = (
    # np.round(x_prime, 10),
    # np.round(y_prime, 10),
    # np.round(z_prime, 10),
    # )

    # Ex = gaussian_E_x(x_prime, y_prime, z_prime, E0, wavelength, w0)
    # Ey = gaussian_E_y(x_prime, y_prime, z_prime, E0, wavelength, w0)
    # Ez = gaussian_E_z(x_prime, y_prime, z_prime, E0, wavelength, w0)
    # return Ex, Ey, Ez
    return x_prime, y_prime, z_prime


wavelength = 1e-6
w0 = wavelength * 5  # breaks down when w0 < wavelength
alpha_by_k = 0.5
kt_by_kz = 0.15  # ratio of transverse to longitudinal wavevector, kz currently set to 2pi/wavelength (in general_bessel_constants)

n1 = 3.9
ep1 = n1 * n1
ep2 = 1.0
radius = 200e-9  # half a micron to one micron
water_permittivity = 80.4
k = 2 * np.pi / wavelength
a0 = (4 * np.pi * 8.85e-12) * (radius ** 3) * ((ep1 - 1) / (ep1 + 2))
a = a0 / (1 - (2 / 3) * 1j * k ** 3 * a0)  # complex form from Chaumet (2000)
#a = a0

#jones_vector = (1,1)
jones_vector = (1 / sqrt(2), (0 + 1j) / sqrt(2))  # change the polarisation of the beam
E0 = 3e6 

n_beams = 1
beam_angles = np.array(
    ((0, 0, 0), (0, 0, 0))
)  # each vector represents each beam, the angles are written as (angle around x axis, y..., z...) in radians
beam_positions = np.array(((0, 0, 0), (0, 0, 0)))

order = 3
lower = -10e-6
upper = -lower

# plot_intensity_xy(100, 100, 1, "general bessel")

# force_distance_plots("gaussian")

#force_map("gaussian")
force_map("general bessel")
#force_map("zeroth bessel")

#~~~testing code~~~
# x = np.linspace(-5e-5, 5e-5, 200)
# y = np.linspace(-5e-5, 5e-5, 200)
# z = np.linspace(-5e-5, 5e-5, 200)
# Ex = np.zeros(200, dtype=complex)
# Ex_prime = np.zeros(200, dtype=complex)
# Ez = np.zeros(200, dtype=complex)
# dum = 0
# for i in range(200):
#     x_prime, y_prime, z_prime = coord_transformation(
#         (x[i], 0, 0), (0, pi/2, 0), (0, 0, 0)
#     )
#     Ex_prime[i] = gaussian_E_x(x_prime, y_prime, z_prime, E0, wavelength, w0)
#     Ex[i] = gaussian_E_x(0, 0, x[i], E0, wavelength, w0)
# plt.plot(x, Ex)
# # plt.plot(x, np.imag(Ex))
# plt.show()

# x = np.linspace(-5e-5, 5e-5, 200)
# y = np.linspace(-5e-5, 5e-5, 200)
# Ex = np.zeros(200, dtype=complex)
# dExdx = np.zeros(200, dtype=complex)
# Ez = np.zeros(200, dtype=complex)
# dEzdx = np.zeros(200, dtype=complex)
# Ey = np.zeros(200, dtype=complex)
# dEydx = np.zeros(200, dtype=complex)
# for i in range(200):
#     Ex[i] = first_bessel_E_x(0, y[i], 0, E0, wavelength)
#     dExdx[i] = first_bessel_dE_xdx(0, y[i], 0, E0, wavelength)
#     Ey[i] = first_bessel_E_y(0, y[i], 0, E0, wavelength)
#     dEydx[i] = first_bessel_dE_ydx(0, y[i], 0, E0, wavelength)
#     Ez[i] = first_bessel_E_z(0, y[i], 0, E0, wavelength)
#     dEzdx[i] = first_bessel_dE_zdx(0, y[i], 0, E0, wavelength)
# plt.plot(x, np.real(np.conjugate(dEzdx) * Ez))
# # plt.plot(x, np.imag(dEzdx))
# plt.show()


# frames = 100
# X, Y, I, z = intensity_xy_animation(50, 50, frames, "gaussian")
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(X, Y, I[0], cmap=cm.coolwarm, linewidth=0, antialiased=False)

# ani = animation.FuncAnimation(fig, update, frames=frames, interval=50)
# # plt.show()
# writer = animation.PillowWriter(fps=20)

# ani.save("gaussian_intensity_plot_w0=2wavl.gif", writer=writer)

# print(gaussian_dE_xdx(0,10,10,10,10,0.5))
