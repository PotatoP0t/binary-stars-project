from phoebe.parameters import ParameterSet as phb
from lightkurve import LightCurve as lk

def saveSimulation(bSystem, filename, filenameFits=None, compact=True):
    """Searches and downloads the MAST data archive for lightcurves. Will save as a .fits file.
    
    Parameters
    ----------
    bSystem : Phoebe ParameterSet
        The phoebe binary simulation.
    filename : str
        The relative or full path to the .phoebe file that will be saved.
    filenameFits: str (optional)
        The relative or full path to the .fits file that will be saved.
        If left blank, this will be the same as filename.
    compact : Bool (optional)
        Decides whether to use compact file-formatting for the .phoebe file.
    """
    
    # Save .phoebe simulation
    bSystem.save(f'{filename}.phoebe', compact=compact)
    
    # Save .fits file
    if not filenameFits: filenameFits=filename
    
    times = bSystem['model@times'].get_value()
    fluxes = bSystem['model@fluxes'].get_value()
    lk(time=times, flux=fluxes).to_fits(f'{filenameFits}_lc.fits', overwrite=True)