import sqlite3, astropy, pandas as pd
from astroquery.mast import Catalogs
from astroquery.gaia import Gaia

def queryGaiaColor(TIC_ID: int, DR=3):
    GAIA_ID = Catalogs.query_object(f'TIC {TIC_ID}', catalog='TIC')['GAIA'][0] # fetch Gaia ID using TIC
    
    if DR == 3: # Query DR3
        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
        
        query = f"""
        SELECT TOP 1
            dr3.source_id AS Gaia_id,
            dr3.bp_rp,
            dr3.phot_g_mean_mag AS Gaia_G
        FROM gaiadr3.gaia_source AS dr3
        JOIN gaiadr3.dr2_neighbourhood AS bridge ON dr3.source_id = bridge.dr3_source_id
        WHERE bridge.dr2_source_id = {GAIA_ID}
        ORDER BY bridge.angular_distance ASC
        """
        
    elif DR == 2: # Query DR2
        Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"
        
        query = f""" 
        SELECT TOP 1
            source_id AS Gaia_id, 
            bp_rp,
            phot_g_mean_mag AS Gaia_G
        FROM gaiadr2.gaia_source
        WHERE source_id = {GAIA_ID}
        """
    
    job = Gaia.launch_job(query)
    df = job.get_results().to_pandas()
    df.insert(1, 'tic_id', [TIC_ID], True)
    return df
