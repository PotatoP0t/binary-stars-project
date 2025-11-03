import lightkurve as lk
from pathlib import Path

def downloadlc(target, path=None, author=None, version=None):
    """Searches and downloads the MAST data archive for lightcurves. Will save as a .fits file.
    
    Parameters
    ----------
    target : str, int
        The target to search for, can include the name of the target or its KIC/EPIC identifier as an integer.
    path : str
        The directory where the file will be stored. Its name will be called target.fits in the specified directory.
        If no specifiec path, will store in current working directory.
    author : str
        The author/pipeline of the lightcurve. Default behavour is it searches all pipelines.
    version : int
        The version of the lightcurve to be stored (starting at index 1). Default behaviour is to download the most recent version (-1).
        If version number is above the total, will download the most recent lightcurve.
        
    Returns
    -------
    lc : LightCurve Object
        Lightcurve of target as a Lightcurve object from Lightkurve
    """
    
    if path: path += f'/{target}_lc.fits'
    else: path = f'/{target}_lc.fits'
    search_query = lk.search_lightcurve(target, author=author)
    if version > len(search_query) or version == None: version = 0
    lc = search_query[version-1].download()
    lc.to_fits(path=path, overwrite=True)
    
    return lc

def loadlc(target, path=None, download=False):
    """Loads lightcurve data into a lightkurve.LightCurve object.
    
    Parameters
    ----------
    target : str, int
        The target lightcurve to load. If stored locally, can specify the filename with file extension instead.
    path : str
        The path to the file or directory of the lightcurve. If not specified or not found, it will automatically download the lightcurve.
    download : bool 
        If the file isn't stored locally, after downloading then store it locally if true.
    """
    
    filepath = Path(f'{path}/{target}')
    if filepath.exists(): # File exists, will assume its a TESS lc
        return lk.read(f'{filepath}')
    else:
        if download: return downloadlc(target, path, 'TESS-SPOC')
        else:
            search_query = lk.search_lightcurve(target, author='TESS-SPOC')
            return search_query[-1].download()
        
    
print(loadlc('TIC 91961_lc.fits', path='Data/TESS'))