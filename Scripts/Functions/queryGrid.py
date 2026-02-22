import sqlite3, astropy, pandas as pd, numpy as np
from astroquery.mast import Catalogs
from astroquery.gaia import Gaia

def queryGaia(TIC_ID: int, DR=3):
    """Queries Gaia DR3 or DR2 for G_BP - G_RP, using TESS ID. It also calculates the error in the colour.
    Additionally, retrieves the Gaia magnitude and Gaia effective temperature.
    
    Input
    -----
    TIC_IC : int
        ID from TIC_v8 catalog
    DR : int
        Choose between DR3 (DR=3) or DR2 (DR=2)
        
    Output
    ------
    df : pandas dataframe
        Dataframe with the following columns: Gaia{}_id, tic_id, Gaia{}_G, bp_rp, Gaia{}_T
        The number 2 or 3 will be appended in the parenthesis if used DR2 or DR3
    
    """
    GAIA_ID = Catalogs.query_object(f'TIC {TIC_ID}', catalog='TIC')['GAIA'][0] # fetch Gaia ID using TIC
    
    if DR == 3: # Query DR3
        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
        
        query = f"""
        SELECT TOP 1
            dr3.source_id AS Gaia3_id,
            dr3.phot_g_mean_mag AS Gaia3_G,
            dr3.teff_gspphot AS Gaia3_T,
            dr3.bp_rp,
            dr3.phot_bp_mean_flux_over_error AS bp_err,
            dr3.phot_rp_mean_flux_over_error AS rp_err
        FROM gaiadr3.gaia_source AS dr3
        JOIN gaiadr3.dr2_neighbourhood AS bridge ON dr3.source_id = bridge.dr3_source_id
        WHERE bridge.dr2_source_id = {GAIA_ID}
        ORDER BY bridge.angular_distance ASC
        """
        
    elif DR == 2: # Query DR2
        Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"
        
        query = f""" 
        SELECT TOP 1
            source_id AS Gaia2_id,
            phot_g_mean_mag AS Gaia2_G,
            teff_val AS Gaia2_T,
            bp_rp,
            phot_bp_mean_flux_over_error AS bp_err,
            phot_rp_mean_flux_over_error AS rp_err
        FROM gaiadr2.gaia_source
        WHERE source_id = {GAIA_ID}
        """
    
    else:
        print("Invalid data release version.")
        return
    
    job = Gaia.launch_job(query)
    table = job.get_results()
    
    # Error calculations
    table['bp_err'] = (2.5 / np.log(10)) / table['bp_err'] 
    table['rp_err'] = (2.5 / np.log(10)) / table['rp_err'] 
    table['bp_rp_err'] = np.sqrt(table['bp_err']**2 + table['rp_err']**2)
    
    table.remove_columns(['bp_err', 'rp_err'])
    df = table.to_pandas()
    df.insert(1, 'TIC_id', [TIC_ID], True)
    return df

def Gaia3ToJohnson(df):
    """
    Converts GaiaDR3 colours to Johnson B-V.
    
    Input
    -----
    df : Pandas Dataframe
        Must contain bp_rp column
        
    Output
    ------
    df : Pandas Dataframe
        Dataframe with B - V appended
    """
        
    # Calculate B - V 
    bv = []
    for i, bprp_i in enumerate(df['bp_rp']):
        bv_coeff = [0.01916, -0.176, 0.5707, -0.7815, 1.575, -0.06483 - bprp_i] # Coefficients for B-V, from Gaia DR3 docs
        roots = np.roots(bv_coeff)
                
        real_roots = roots[np.isclose(roots.imag, 0)].real # Pull real roots
        if len(real_roots) > 0:
            index = np.abs(real_roots - df['bp_rp'][i]).argmin() # Multple roots, b-v is closest to bp-rp
            bv.append(real_roots[index]) 
        else:
            bv.append(np.nan)
        
    df['b_v'] = bv
    df['b_v_err'] = 0.0659 # Error from conversion >> error in Gaia BP-RP color, so will disregard color error propagation for convinience
    return df

def queryPrimary(df_target, method='b-v', filepath='Data/binarystargrid.db'):
    """Query the grid of simulations based on colour and return back a dataframe with a subset of the grid with the closest match to primary star.
    
    Input
    -----
    df_target : Pandas Dataframe
        Dataframe for the target system.
    
    method : str
        What method to use, either 'B-V' (DR3) or 'BP-RP' (DR2)
    
    filepath : str
        Path to database
        
    Output
    ------
    df_grid : Pandas Dataframe
        Subgrid of simulations with closest primary star to target system. 
    """
    
    conn = sqlite3.connect(filepath)
    
    if method == 'B-V':
        target_b_v = df_target['b_v'].iloc[0]
        query = """
        SELECT * FROM "F5-K4"
        WHERE ABS("Primary B-V" - ?) = (SELECT MIN (ABS("Primary B-V" - ?)) FROM "F5-K4")
        """
        
        df_grid = pd.read_sql(query, conn, params=(target_b_v, target_b_v))
        conn.close()
        return df_grid 
    elif method == 'BP-RP':
        target_bp_rp = float(df_target['bp_rp'].iloc[0])
        query = """
            SELECT * FROM "F5-K4"
            WHERE ABS("Primary BP-RP" - ?) <= (
                SELECT MIN(ABS("Primary BP-RP" - ?)) FROM "F5-K4"
            )
        """
        
        df_grid = pd.read_sql(query, conn, params=(target_bp_rp, target_bp_rp))
        conn.close()
        return df_grid
    else:
        print("No valid query target.")
        return

df = queryGaia(288434781, 2)
df = Gaia3ToJohnson(df)
print(df)
df_grid = queryPrimary(df, 2)
print(df_grid)