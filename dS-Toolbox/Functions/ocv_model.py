import numpy as np


def OCVfromSOC_function(soc, temp, model):
    """
    Computes the fully rested open-circuit voltage for a particular
    state of charge and temperature.

    Parameters
    ----------
    soc : scalar or array_like
        State of charge values
    temp : scalar or array_like
        Temperature values
    model : dict-like
        Must contain:
            model["SOC"]
            model["OCV0"]
            model["OCVrel"]

    Returns
    -------
    ocv : ndarray or scalar
        Open circuit voltage values (same shape as soc input)
    """

    # Convert to numpy arrays
    soc = np.asarray(soc)
    original_shape = soc.shape
    soccol = soc.flatten()

    SOC = np.asarray(model["SOC"]).flatten()
    OCV0 = np.asarray(model["OCV0"]).flatten()
    OCVrel = np.asarray(model["OCVrel"]).flatten()

    # Temperature handling
    if np.isscalar(temp):
        tempcol = np.full_like(soccol, temp, dtype=float)
    else:
        temp = np.asarray(temp)
        tempcol = temp.flatten()
        if tempcol.shape != soccol.shape:
            raise ValueError(
                '"soc" and "temp" must have same number of elements '
                'or "temp" must be scalar.'
            )

    diffSOC = SOC[1] - SOC[0]  # assume uniform spacing
    ocv = np.zeros_like(soccol, dtype=float)

    # Index masks
    I1 = soccol <= SOC[0]
    I2 = soccol >= SOC[len(SOC)-1]
    I3 = (soccol > SOC[0]) & (soccol < SOC[len(SOC)-1])
    I6 = np.isnan(soccol)

    # --- Low-end extrapolation ---
    if np.any(I1):
        dv = (
            (OCV0[1] + tempcol * OCVrel[1])
            - (OCV0[0] + tempcol * OCVrel[0])
        )
        ocv[I1] = (
            (soccol[I1] - SOC[0]) * dv[I1] / diffSOC
            + OCV0[0]
            + tempcol[I1] * OCVrel[0]
        )

    # --- High-end extrapolation ---
    if np.any(I2):
        dv = (
            (OCV0[len(OCV0)-1] + tempcol * OCVrel[len(OCVrel)-1])
            - (OCV0[len(OCV0)-2] + tempcol * OCVrel[len(OCVrel)-2])
        )
        ocv[I2] = (
            (soccol[I2] - SOC[len(SOC)-1]) * dv[I2] / diffSOC
            + OCV0[len(OCV0)-1]
            + tempcol[I2] * OCVrel[len(OCVrel)-1]
        )

    # --- Linear interpolation (manual, like MATLAB) ---
    if np.any(I3):
        I4 = (soccol[I3] - SOC[0]) / diffSOC
        I5 = np.floor(I4).astype(int)
        I45 = I4 - I5
        omI45 = 1 - I45

        ocv[I3] = (
            OCV0[I5] * omI45
            + OCV0[I5 + 1] * I45
        )

        ocv[I3] += tempcol[I3] * (
            OCVrel[I5] * omI45
            + OCVrel[I5 + 1] * I45
        )

    # Replace NaN SOCs with zero
    ocv[I6] = 0

    return ocv.reshape(original_shape)