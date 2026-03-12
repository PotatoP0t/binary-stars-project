import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from tqdm import tqdm

from scipy.signal import fftconvolve
from scipy.linalg import norm
from scipy.optimize import minimize, least_squares

'''
=== Example code for plotting test graphs ===

plt.plot(phi, convolve, "ro")
plt.tick_params(axis='both', labelsize=12)
plt.minorticks_on()
plt.xlabel("$\\phi$", fontsize=12)
plt.ylabel("SNR", fontsize=12)
plt.xlim(0, 1)
plt.show()

'''


# Returns current time in seconds
def get_time():
    dt = datetime.now()
    time = dt.hour*3600 + dt.minute*60 + dt.second

    return time



# z is the rescaled phase used for our eclipse model V = V(z, beta)
def z(phi, phi0, D):
    return ((phi-phi0) / (D*0.5))

# V is the eclipse model, returns 0 if |z|>1
def V(phi, beta, D, phi0, normalise=True):

    z_val = z(phi, phi0, D)
    model = np.zeros_like(z_val)

    # Calculates V at all indexes where the condition is met
    model_condition = np.abs(z_val) <= 1
    ensure_positive = 1 - np.abs(z_val[model_condition]) ** beta
    ensure_positive = np.clip(ensure_positive, 0, None)
    model[model_condition] = -1*(ensure_positive)**1.5

    # Normalise so that a larger width != larger matched filter score
    # normalise=False during minimise function, i.e. only run for matched filter
    if normalise:
        model_norm = norm(model)
        if model_norm > 0:
            model /= model_norm

    return model



# Returns phase diagram data for a light curve
def fold(time, flux, P):

    # Calculates the phase of each time point
    phi = np.asarray(((time - time[0]) / P) % 1)
    
    # Orders the data by ascending phase
    i_ord = np.argsort(phi)
    phi_ord = phi[i_ord]
    flux_ord = np.asarray(flux[i_ord])

    return(phi_ord, flux_ord, i_ord)



# Calculates a harmonic ratio using Fourier coefficients
def harmonic_ratio(phi, flux):

    # Fourier coefficients for n=1,2,4
    a1 = (2/len(phi)) * np.sum(flux * np.cos(2 * np.pi * phi))
    b1 = (2/len(phi)) * np.sum(flux * np.sin(2 * np.pi * phi))
    a2 = (2/len(phi)) * np.sum(flux * np.cos(4 * np.pi * phi))
    b2 = (2/len(phi)) * np.sum(flux * np.sin(4 * np.pi * phi))
    a4 = (2/len(phi)) * np.sum(flux * np.cos(8 * np.pi * phi))
    b4 = (2/len(phi)) * np.sum(flux * np.sin(8 * np.pi * phi))

    # Fourier amplitudes for n=1,2
    # A1 big for 1 repeating feature per period
    A1 = np.sqrt(a1**2 + b1**2)
    # A2 big for 2 repeating features per period
    A2 = np.sqrt(a2**2 + b2**2)
    # A4 big for 4 repeating features per period
    A4 = np.sqrt(a4**2 + b4**2)

    return A1/A2, A4/A2




'''
==================
=   Estimation   =
==================
'''

def find_param_candidates(time, flux, period_t, width_t):

    # Initialising variables
    # Signal-to-noise ratio peak for each trial period (and width for the inner loop)
    snr_peak = np.zeros(len(period_t))
    snr_peak_per_width = np.zeros(len(width_t))
    # Final score for each trial period
    # Shape (2, len(period_t)) for [[peak1, peak2, ...], [width at peak1, width at peak2, ...]]
    score = np.zeros((2, len(period_t)))
    #percentage_complete = 0

    for P_ind in tqdm(range(0, len(period_t))):

        # Phase fold the light curve
        phi, f_flux, _ = fold(time, flux, period_t[P_ind])

        for D_ind in range(0, len(width_t)):

            # Eclipse model values at each phi value in this phase diagram
            # beta = 1.5 (reasonable starting shape), phi_0 = 0.5 (centered eclipse)
            model = V(phi, 1.5, width_t[D_ind], 0.5)
            # Compare folded flux to model
            convolve = fftconvolve(f_flux, model, mode="same")
            # Store SNR peak for this trial width
            snr_peak_per_width[D_ind] = np.max(convolve)

        # Store SNR peak for this trial period
        snr_peak[P_ind] = np.max(snr_peak_per_width)

        # Harmonic ratio A2/A1 approx. 0 for P/2, and > 0 for P
        # Creates a score that considers both the eclipse features and the periodicity
        # Added a punishment for 2P by dividing by 1 + A4/A2
        harm_rat_1to2, harm_rat_4to2 = harmonic_ratio(phi, f_flux)
        score[0, P_ind] = snr_peak[P_ind] / ((1 + harm_rat_1to2) * (1 + harm_rat_4to2))
        # Stores the width associated with the peak used to calculate the score
        max_i = np.argmax(snr_peak_per_width)
        score[1, P_ind] = width_t[max_i]


    # This section was replaced by tqdm loading bar
    #     # Completion tracker for stdout
    #     if P_ind !=0 and P_ind % ((len(period_t))/5) == 0:
    #         percentage_complete += 20
    #         print(f"Finding P candidates: {percentage_complete}% complete")

    # print("Finding P candidates: 100% complete")

    # Picks out two periods corresponding to two highest scores and what widths those were calculated at
    score_ord_idx = np.argsort(score[0])
    P_candidate_1 = period_t[score_ord_idx[-1]]
    width_P1 = score[1, score_ord_idx[-1]]
    # P_candidate_2 = period_t[score_ord_idx[-2]]
    # width_P2 = score[1, score_ord_idx[-2]]

    # print(f"\nP estimate 1: {P_candidate_1} days at width: {width_P1:.3f}")
    # print(f"P estimate 2: {P_candidate_2} days at width: {width_P2:.3f}")

    return P_candidate_1, width_P1




'''
====================
=   Optimisation   =
====================
'''

# baseline + primary eclipse + secondary eclipse
def full_model(phi, flux, std_dev, beta_1, beta_2, D_1, D_2, phi_1, phi_2, bl):

    # Horizontal line at the baseline
    baseline = np.zeros_like(phi)
    baseline.fill(bl)

    # Eclipse model at depth 1
    shape_1 = V(phi, beta_1, D_1, phi_1, normalise=False)
    shape_2 = V(phi, beta_2, D_2, phi_2, normalise=False)
    # Used to find chi squ, or in this case, depths and their errors
    weights = 1 / std_dev**2

    # Depths and complete eclipse models
    A_1 = np.sum(shape_1 * flux * weights) / np.sum(shape_1**2 * weights)
    A_2 = np.sum(shape_2 * flux * weights) / np.sum(shape_2**2 * weights)
    eclipse_1 = A_1 * shape_1
    eclipse_2 = A_2 * shape_2

    # Depth errors
    model = baseline + eclipse_1 + eclipse_2
    res = flux - model
    sigma_res = np.sqrt(np.sum((res/std_dev)**2) / (len(flux)-10))
    A1_err  = sigma_res / np.sqrt(np.sum(shape_1**2 * weights))
    A2_err  = sigma_res / np.sqrt(np.sum(shape_2**2 * weights))

    return(model, [A_1, A1_err], [A_2, A2_err])



###################
# Different Chi Squared functions for different fixed variables
###################

# Chi squared function setup for eclipse params to be optimised
def chi_squ_shape(params, P, time, flux, flux_err, return_res=False, e1=True, e2=True):

    beta_1, beta_2, D_1, D_2, phi_1, phi_2, bl = params
    phi, f_flux, i_ord = fold(time, flux, P)
    std_dev = flux_err[i_ord]
    model, _, _ = full_model(phi, f_flux, std_dev, *params)

    # Punish unphysical results
    if not np.all(np.isfinite(model)):
        return 1e30
    
    if not np.all([
        beta_1 > 0,
        beta_2 > 0,
        1e-4 < D_1 < 0.5,
        1e-4 < D_2 < 0.5,
        0 < phi_1 < 1,
        0 < phi_2 < 1,
        P > 0
    ]):
        return 1e30
    

    if e1 and not e2:
        mask = np.abs(z(phi, phi_1, D_1)) <= 1
    elif e2 and not e1:
        mask = np.abs(z(phi, phi_2, D_2)) <= 1
    else:
        mask = np.ones_like(phi, dtype=bool)

    # Use this function to return residuals instead for width errors
    if return_res:
        return (f_flux - model) / std_dev
    
    return np.sum(((f_flux[mask] - model[mask]) / std_dev[mask])**2)

# Chi squared function setup for P to be optimised
def chi_squ_period(period, params, time, flux, flux_err):

    P = period[0]
    phi, f_flux, i_ord = fold(time, flux, P)
    std_dev = flux_err[i_ord]
    model, _, _ = full_model(phi, f_flux, std_dev, *params)

    # Punish unphysical results
    if not np.all(np.isfinite(model)):
        return 1e30
    
    if P <= 0:
        return 1e30

    # If physical, return chi squared
    return np.sum(((f_flux - model) / std_dev)**2)



def optimise_params(P, D, time, flux, flux_err):

    phi, f_flux, _ = fold(time, flux, P)
    
    # Initial parameters
    beta_1 = 1.5    # Sensible starting shape
    beta_2 = 1.5
    D_1 = D    # Best trial width
    D_2 = D
    phi_1 = phi[np.argmin(f_flux)]
    phi_2 = (phi_1 + 0.5) % 1
    bl = 0      # Baseline removed so ~0

    init_params = [beta_1, beta_2, D_1, D_2, phi_1, phi_2, bl]

    # Find optimal eclipse params with fixed P
    result1 = minimize(chi_squ_shape,
                    init_params, 
                    args=(P, time, flux, flux_err), 
                    method="Nelder-Mead",
                    options={"maxfev": 50000})
    
    print(f"\nEclipse parameters: {result1.message}")

    # Find optimal P with fixed eclipse params
    result2 = minimize(chi_squ_period,
                       [P],
                       args=(result1.x, time, flux, flux_err),
                       method="Nelder-Mead",
                       options={"maxfev": 50000})
    
    print(f"Period: {result2.message}")

    # Find depths from the model
    beta_1, beta_2, D_1, D_2, phi_1, phi_2, bl = result1.x
    phi, f_flux, i_ord = fold(time, flux, result2.x[0])
    _, A_1, A_2 = full_model(phi, f_flux, flux_err[i_ord], beta_1, beta_2, D_1, D_2, phi_1, phi_2, bl)

    P = result2.x[0]

    result3 = least_squares(chi_squ_shape,
                            result1.x,
                            args=(P, time, flux, flux_err, True))
    
    J = result3.jac
    res = result3.fun
    cov = np.sum(res**2)/(len(res)-len(result3.x)) * np.linalg.inv(J.T @ J)
    param_errs = np.sqrt(np.diag(cov))
    sigma_D1 = param_errs[2]
    sigma_D2 = param_errs[3]


    #########################
    # Finding 1 std dev of P
    #########################
    print("\nCalculating P error...")
    
    P_test_vals = np.linspace(P-0.01, P+0.01, 100)
    chi_vals = []

    for P_test in P_test_vals:
        chi_vals.append(chi_squ_shape(result1.x, P_test, time, flux, flux_err))
    
    # Where the change in chi squared = 1
    chi_min = np.min(chi_vals)
    indices = np.where(chi_vals <= chi_min+1)[0]
    close_P = []

    for i in indices:
        close_P.append(P_test_vals[i])

    # Find 1 std dev by seeing what P corresponds to chi squared +- 1 
    P_min = np.min(close_P)
    P_max = np.max(close_P)
    sigma_P = np.max([np.abs(P_min - P), np.abs(P_max - P)])

    errors = [sigma_P, sigma_D1, sigma_D2, A_1[1], A_2[1]]

    # Period, depths, other variables, errors, chi squared
    return(result2.x[0], [A_1[0], A_2[0]], result1.x, errors, result2.fun)
    



'''
============
=   MAIN   =
============
'''

def main(target, author=None, idx=-1, plot=False):

    print(f"\nCalculating parameters for {target}\n")

    # Finds and downloads a specific lightcurve
    search_result = lk.search_lightcurve(target, author=author)[idx]
    lc = search_result.download()

    # Defining initial trial variables and parameters
    # Period, 5 to 20 in 0.001 increments [days]
    period_t = np.linspace(0.001, 20, num=20000)
    # Eclipse width (logspace to favour narrow widths) [fraction of period]
    width_t = np.logspace(np.log10(0.001), np.log10(0.3), num=8)

    # Setup data
    lc = lc.remove_nans()
    time = lc.time.value
    # Normalises then removes baseline from the data
    baseline = np.median(lc.flux.value)
    flux = lc.flux.value / baseline
    flux -= 1
    flux_err = lc.flux_err.value

    # Use a matched filter method to find an estimate for the period, P, and width, D
    P_estimate, D_estimate = find_param_candidates(time, flux, period_t, width_t)
    # Optimise P and model parameters to minimise the chi squared between the model and data
    best_period, best_depths, best_params, param_errs, chi_squared = optimise_params(P_estimate, D_estimate, time, flux, flux_err)

    P = best_period
    sigma_P = param_errs[0]
    beta_1, beta_2, D_1, D_2, phi_1, phi_2, bl = best_params
    sigma_D1 = param_errs[1]
    sigma_D2 = param_errs[2]
    A_1 = best_depths[0]
    sigma_A1 = param_errs[3]
    A_2 = best_depths[1]
    sigma_A2 = param_errs[4]

    print(f"\nPeriod: {P} +- {sigma_P} days, Chi squared: {chi_squared}")
    print(f"Width 1: {D_1} +- {sigma_D1}, Width 2: {D_2} +- {sigma_D2}")
    # print(f"Location 1: {phi_1}, Location 2: {phi_2}")
    print(f"Depth 1: {A_1} +- {sigma_A1}, Depth 2: {A_2} +- {sigma_A2}")
    # print(f"Shape 1: {beta_1}, Shape 2: {beta_2}")
    # print(f"Baseline: {bl}")
    print("")

    if plot:
        # Plot data and final model
        phi, f_flux, i_ord = fold(time, flux, P)
        model, _, _ = full_model(phi, f_flux, flux_err[i_ord], *best_params)

        fig, [ax0, ax1, ax2] = plt.subplots(3)

        ax0.plot(time, flux)
        ax0.tick_params(axis='both', labelsize=12)
        ax0.minorticks_on()

        ax1.plot(phi, f_flux, "ro")
        ax1.tick_params(axis='both', labelsize=12)
        ax1.minorticks_on()

        ax2.plot(phi, model, "b")
        ax2.tick_params(axis='both', labelsize=12)
        ax2.minorticks_on()

        fig.suptitle(f"{target}")
        fig.text(0.01, 0.33, "$\\Delta$ Normalised Flux", va="center", rotation="vertical", fontsize=12)
        plt.xlabel("$\\phi$", fontsize=12)
        plt.show()

    return(P, best_depths, best_params, param_errs, chi_squared)