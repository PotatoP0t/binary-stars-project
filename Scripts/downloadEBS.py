import Functions.queryLightcurve as queryLightcurve
from multiprocessing import Pool
import pandas as pd, numpy as np

def subprocess(tess_ids):
    for tess_id in tess_ids: queryLightcurve.downloadlc('TIC ' + str(tess_id), 'Data', 'TESS-SPOC')

def main():
    df = pd.read_csv('Data/TESS EBS Catalogue.csv')
    
    # Control values
    # --------------
    period = 5 # Searches for EBS in catalogue with period higher
    prim_sec_depth_diff = .1 # Searches for EBS with % difference between primary and secondary depth less than this
    prim_depth = (.4, .6) # Searches for EBS with a primary depth between [(lower), (higher)]

    # Candidates with higher periods are likely to be detached systems - safe for assuming main sequence stars as they probably haven't had any interactions between them to mess up stellar evolution
    df = df.query(f'period > {period}')
    
    # Look for binaries where the difference between primary and secondary difference is smaller
    df['identical'] = np.where((abs(df['prim_depth_2g'] - df['sec_depth_2g'])/df['prim_depth_2g'] < prim_sec_depth_diff), True, False)
    df = df.query('identical == True')
    
    # Look for binaries where the primary depth is great to avoid partial eclipses
    df = df.query(f'prim_depth_2g > {prim_depth[0]}')
    df = df.query(f'prim_depth_2g < {prim_depth[1]}')
     
    tess_ids = df['tess_id']
    
    assert input(f'Candidates found: {len(tess_ids)}\nDownload (y/[n])?: ') == 'y'
    
    # Just some parallel processing to make downloading files easier
    n_core = 10
    tess_ids = np.array_split(tess_ids, n_core)
    
    with Pool(processes=n_core) as pool:
        pool.map(subprocess, tess_ids)


if __name__ == '__main__':
    main()