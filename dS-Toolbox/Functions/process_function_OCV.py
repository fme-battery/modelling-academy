import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from Functions.ocv_model import OCVfromSOC_function  # previously converted core function
import os

def processOCV(data, cellID, minV, maxV, savePlots=False):
    """
    Estimates battery SOC window from charge/discharge capacities
    and computes OCV0/OCVrel and SOC0/SOCrel profiles over multiple temperatures.
    
    Parameters
    ----------
    data : list of dicts
        Each dict represents a temperature dataset with keys: 'temp', 'script1', 'script3'.
        Each script is a dict with 'step', 'voltage', 'chgAh', 'disAh'.
    cellID : str
        Cell identifier
    minV : float
        Minimum voltage for plotting
    maxV : float
        Maximum voltage for plotting
    savePlots : bool
        If True, save figures to disk.
        
    Returns
    -------
    model : dict
        Contains SOC, OCV, OCV0, OCVrel, SOC0, SOCrel, eta, Q, and other metadata.
    """

    # --- Extract temperatures ---
    filetemps = np.array([d['temp'] for d in data])
    numtemps = len(filetemps)

    # Must have 25°C dataset
    idx25 = np.where(filetemps == 25)[0]
    if len(idx25) == 0:
        raise ValueError("Must have a test at 25°C")
    idx25 = idx25[0]
    not25 = np.where(filetemps != 25)[0]

    SOC = np.arange(0, 1.001, 0.001)  # SOC grid

    filedata = [{} for _ in range(numtemps)]
    eta = np.zeros(numtemps)
    Q = np.zeros(numtemps)

    # ----------------------------
    # Process 25°C data first
    # ----------------------------
    k = idx25
    dataS1V = np.array(data[k]['script1']['voltage'])
    dataS1disAh = np.array(data[k]['script1']['disAh'])
    dataS1chgAh = np.array(data[k]['script1']['chgAh'])
    dataS1step = np.array(data[k]['script1']['step'])

    dataS3V = np.array(data[k]['script3']['voltage'])
    dataS3disAh = np.array(data[k]['script3']['disAh'])
    dataS3chgAh = np.array(data[k]['script3']['chgAh'])
    dataS3step = np.array(data[k]['script3']['step'])

    totDisAh = dataS1disAh[-1] + dataS3disAh[-1]
    totChgAh = dataS1chgAh[-1] + dataS3chgAh[-1]
    
    eta25 = totDisAh / totChgAh
    eta[k] = eta25

    #print (eta25)
    #print (eta[k])

    # Adjust charge Ah for coulombic efficiency
    dataS1chgAh = dataS1chgAh * eta25
    dataS3chgAh = dataS3chgAh * eta25

    Q25 = dataS1disAh[-1] - dataS1chgAh[-1]
    Q[k] = Q25

    # --- Discharge processing ---
    step1 = dataS1step
    indD = np.where(step1 == 2)[0]  # slow discharge step
    voltD = dataS1V
    IR1Da = voltD[indD[0]-1] - voltD[indD[0]]
    IR2Da = voltD[indD[-2]] - voltD[indD[-1]]

    step3 = dataS3step
    voltC = dataS3V
    indC = np.where(step3 == 2)[0]  # slow charge step
    IR1Ca = voltC[indC[0]] - voltC[indC[0]-1]
    IR2Ca = voltC[indC[-1]] - voltC[indC[-2]]

    IR1D = min(IR1Da, 2*IR2Ca)
    IR2D = min(IR2Da, 2*IR1Ca)
    IR1C = min(IR1Ca, 2*IR2Da)
    IR2C = min(IR2Ca, 2*IR1Da)

    # Linear resistance blending
    blendD = np.linspace(0, 1, len(indD))
    IRblendD = IR1D + (IR2D - IR1D) * blendD
    disV = voltD[indD] + IRblendD
    disV = disV + np.arange(1, len(disV)+1) * np.finfo(float).eps
    disZ = 1 - dataS1disAh[indD] / Q25
    disZ = disZ + (1 - disZ[0])
    disZ = np.array(disZ, dtype=float)
    disZ = disZ + np.arange(1, len(disZ)+1) * np.finfo(float).eps
   

    blendC = np.linspace(0, 1, len(indC))
    IRblendC = IR1C + (IR2C - IR1C) * blendC
    chgV = voltC[indC] - IRblendC
    chgV = chgV + np.arange(1, len(chgV)+1) * np.finfo(float).eps
    chgZ = dataS3chgAh[indC] / Q25
    chgZ = chgZ - chgZ[0]
    chgZ = np.array(chgZ, dtype=float)
    chgZ = chgZ + np.arange(1, len(chgZ)+1) * np.finfo(float).eps

    # Compute deltaV50 for i*R compensation
    deltaV50 = interp1d(chgZ, chgV)(0.5) - interp1d(disZ, disV)(0.5)
    # Split data around 50% SOC
    vChg = chgV[chgZ > 0.5] - chgZ[chgZ > 0.5] * deltaV50
    zChg = chgZ[chgZ > 0.5]
    vDis = disV[disZ < 0.5] + (1 - disZ[disZ < 0.5]) * deltaV50
    zDis = disZ[disZ < 0.5]

    # Flip discharge arrays to align
    vDis = vDis[::-1]
    zDis = zDis[::-1]


    # Interpolate raw OCV
    filedata[k]['rawocv'] = interp1d(np.concatenate([zChg, zDis]), np.concatenate([vChg, vDis]), kind='linear', bounds_error=False, fill_value="extrapolate")(SOC)
    filedata[k]['temp'] = data[k]['temp']
    filedata[k]['disZ'] = disZ
    filedata[k]['disV'] = voltD[indD]
    filedata[k]['chgZ'] = chgZ
    filedata[k]['chgV'] = voltC[indC]

    # ----------------------------
    # Process other temperatures
    # ----------------------------
    for k in not25:

        dataS1V = np.array(data[k]['script1']['voltage'])
        dataS1disAh = np.array(data[k]['script1']['disAh'])
        dataS1chgAh = np.array(data[k]['script1']['chgAh'])
        dataS1step = np.array(data[k]['script1']['step'])

        dataS3V = np.array(data[k]['script3']['voltage'])
        dataS3disAh = np.array(data[k]['script3']['disAh'])
        dataS3chgAh = np.array(data[k]['script3']['chgAh'])
        dataS3step = np.array(data[k]['script3']['step'])

        totDisAh = dataS1disAh[-1] + dataS3disAh[-1]
        totChgAh = dataS1chgAh[-1] + dataS3chgAh[-1]
        eta[k] = totDisAh / totChgAh

        dataS1chgAh = dataS1chgAh * eta[k]
        dataS3chgAh = dataS3chgAh * eta[k]

        Q[k] = dataS1disAh[-1] - dataS1chgAh[-1]

        # Discharge
        step1 = dataS1step
        voltD = dataS1V
        indD = np.where(step1 == 2)[0]
        IR1D = voltD[indD[0]-1] - voltD[indD[0]]
        IR2D = voltD[indD[len(indD)-1]-1] - voltD[indD[len(indD)-1]]

        # Charge
        step3 = dataS3step
        voltC = dataS3V
        indC = np.where(step3 == 2)[0]
        IR1C = voltC[indC[0]] - voltC[indC[0]-1]
        IR2C = voltC[indC[len(indC)-1]] - voltC[indC[len(indC)-2]]

        # Bound IRs
        IR1D = min(IR1D, 2*IR2C)
        IR2D = min(IR2D, 2*IR1C)
        IR1C = min(IR1C, 2*IR2D)
        IR2C = min(IR2C, 2*IR1D)

        # Linear blending
        blendD = np.linspace(0, 1, len(indD))
        IRblendD = IR1D + (IR2D - IR1D) * blendD
        disV = voltD[indD] + IRblendD
        disV = disV + np.arange(1, len(disV)+1) * np.finfo(float).eps
        disZ = 1 - dataS1disAh[indD] / Q25
        disZ = disZ + (1 - disZ[0])
        disZ = np.array(disZ, dtype=float)
        disZ = disZ + np.arange(1, len(disZ)+1) * np.finfo(float).eps
        filedata[k]['disZ'] = disZ
        filedata[k]['disV'] = voltD[indD]

        blendC = np.linspace(0, 1, len(indC))
        IRblendC = IR1C + (IR2C - IR1C) * blendC
        chgV = voltC[indC] - IRblendC
        chgV = chgV + np.arange(1, len(chgV)+1) * np.finfo(float).eps
        chgZ = dataS3chgAh[indC] / Q25
        chgZ = chgZ - chgZ[0]
        chgZ = np.array(chgZ, dtype=float)
        chgZ = chgZ + np.arange(1, len(chgZ)+1) * np.finfo(float).eps
        filedata[k]['chgZ'] = chgZ
        filedata[k]['chgV'] = voltC[indC]
        deltaV50 = interp1d(chgZ, chgV)(0.5) - interp1d(disZ, disV)(0.5)
        
        vChg = chgV[chgZ > 0.5] - chgZ[chgZ > 0.5] * deltaV50
        zChg = chgZ[chgZ > 0.5]
        vDis = disV[disZ < 0.5] + (1 - disZ[disZ < 0.5]) * deltaV50
        zDis = disZ[disZ < 0.5]


        vDis = vDis[::-1]
        zDis = zDis[::-1]
        filedata[k]['rawocv'] = interp1d(np.concatenate([zChg, zDis]), np.concatenate([vChg, vDis]), kind='linear', bounds_error=False, fill_value="extrapolate")(SOC)
        filedata[k]['temp'] = data[k]['temp']

    # ----------------------------
    # Linear least-squares to determine OCV0 and OCVrel
    # ----------------------------
    Vraw = []
    temps_kept = []
    for k in range(numtemps):
        if filedata[k]['temp'] > 0:
            Vraw.append(filedata[k]['rawocv'])
            temps_kept.append(filedata[k]['temp'])
    Vraw = np.array(Vraw)
    temps_kept = np.array(temps_kept)

    H = np.column_stack([np.ones(len(temps_kept)), temps_kept])
    OCV0 = np.zeros_like(SOC)
    OCVrel = np.zeros_like(SOC)
    for i, _ in enumerate(SOC):
        X, _, _, _ = np.linalg.lstsq(H, Vraw[:, i], rcond=None)
        OCV0[i] = X[0]
        OCVrel[i] = X[1]
    # ----------------------------
    # SOC0 and SOCrel
    # ----------------------------
    z = np.arange(0, 1.001, 0.001)
    v = np.arange(minV - 0.01, maxV + 0.01, 0.01)
    socs = []

    for T in filetemps:
        v1 = OCVfromSOC_function(z, T, {'SOC': SOC, 'OCV0': OCV0, 'OCVrel': OCVrel})
        socs.append(np.interp(v, v1, z))
    socs = np.array(socs)
    H_soc = np.column_stack([np.ones(len(filetemps)), filetemps])
    SOC0 = np.zeros_like(v)
    SOCrel = np.zeros_like(v)
    for i in range(len(v)):
        X, _, _, _ = np.linalg.lstsq(H_soc, socs[:, i], rcond=None)
        SOC0[i] = X[0]
        SOCrel[i] = X[1]

    # ----------------------------
    # Assemble model dict
    # ----------------------------
    model = {
        "OCV0": OCV0,
        "OCVrel": OCVrel,
        "SOC": SOC,
        "Sapprox": OCVrel,
        "OCV": v,
        "SOC0": SOC0,
        "SOCrel": SOCrel,
        "OCVeta": eta,
        "OCVQ": Q,
        "name": cellID,
        "OCVaprox": v1,
        "SOCaprox": z,
        "filedata": filedata
    }
    # ----------------------------
    # Plot OCV curves
    # ----------------------------
    cols = 2
    rows = int(np.ceil(numtemps / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    axes = axes.flatten()

    for k in range(numtemps):
        ax = axes[k]
        temp_k = filedata[k]['temp']
        raw_ocv = filedata[k]['rawocv']
        ax.plot(SOC, OCVfromSOC_function(SOC, temp_k, model), 'b', lw=1.2, label='Model prediction')
        ax.plot(SOC, raw_ocv, 'r', lw=1.2, label='Approximate OCV from data')
        ax.plot(np.array(filedata[k]['disZ']), np.array(filedata[k]['disV']), 'k--', lw=1, label='Measured data')
        ax.plot(np.array(filedata[k]['chgZ']), np.array(filedata[k]['chgV']), 'k--', lw=1)
        ax.set_xlabel('SOC (-)')
        ax.set_ylabel('OCV (V)')
        ax.set_ylim([minV-0.1, maxV+0.1])
        ax.set_xlim([0, 1])
        ax.set_title(f'{cellID} OCV relationship at temp = {temp_k}°C')
        err = raw_ocv - OCVfromSOC_function(SOC, temp_k, model)
        rmserr = np.sqrt(np.mean(err**2))
        ax.text(0.1, maxV-0.15, f'RMS error = {rmserr*1000:.1f} mV', fontsize=10)
        if k == 0:
            ax.legend(loc='lower right')

    plt.tight_layout()

    if savePlots:
        os.makedirs('OCV_FIGURES', exist_ok=True)
        plt.savefig(f'OCV_FIGURES/{cellID}_OCV_grid.png', dpi=300)

    return model
