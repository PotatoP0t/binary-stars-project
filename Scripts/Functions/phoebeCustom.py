from phoebe.parameters import ParameterSet as phb
from lightkurve import LightCurve as lk
import numpy as np
from copy import copy

def saveSimulation(bSystem, filename, filenameFits=None, compact=True):
    """All in one save function for PHOEBE simulations. 
    Saves both as a .phoebe file (to load into PHOEBE) and .fits files for each model (to load into Lightkurve).
    
    Parameters
    ----------
    bSystem : PHOEBE Bundle
        The phoebe binary simulation.
    filename : str
        The relative or full path to the .phoebe file that will be saved.
    filenameFits: str (optional)
        The relative or full path to the .fits file that will be saved.
        If left blank, this will be the same as filename.
    compact : Bool (optional, default=True)
        Decides whether to use compact file-formatting for the .phoebe file.
    """
    
    # Save .phoebe simulation
    bSystem.save(f'{filename}.phoebe', compact=compact)
    
    # Save .fits file
    if not filenameFits: filenameFits=filename
    
    models = bSystem.models
    if len(models) != 1: # Have multiple models, save each
        for model in models:
            times = bSystem[f'{model}@model@times'].get_value()
            fluxes = bSystem[f'{model}@model@fluxes'].get_value()
            lk(time=times, flux=fluxes).to_fits(f'{filenameFits}_{model}_lc.fits', overwrite=True)
    else: # Only one model, slightly different formatting
        times = bSystem['model@times'].get_value()
        fluxes = bSystem['model@fluxes'].get_value()
        lk(time=times, flux=fluxes).to_fits(f'{filenameFits}_lc.fits', overwrite=True)
        
def normaliseFluxes(bSystem):
    """
    Normalises fluxes in PHOEBE simulations for easier comparison to real data.
    Returns a Bundle instance with normalised fluxes.
    
    Parameters
    ----------
    bSystem : PHOEBE Bundle
        The PHOEBE binary simulation
        
    Returns
    -------
    bSystemNormalize : PHOEBE Bundle
        The normalised PHOEBE binary simulation
    """
    
    bSystemNormalize = copy(bSystem)
    
    for model in bSystem.models:
        flux = bSystem[f'{model}@fluxes'].get_value()
        medium_flux = np.nanmedian(flux)
        bSystemNormalize[f'{model}@fluxes'].set_value(value=flux/medium_flux, ignore_readonly=True)
        
    return bSystemNormalize