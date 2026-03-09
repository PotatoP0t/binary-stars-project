import sqlite3, pandas as pd, numpy as np
from astroquery.mast import Catalogs
from astroquery.gaia import Gaia

def queryGaia(TIC_IDs, DR=3):
    """Queries Gaia DR3 or DR2 for G_BP - G_RP, using TESS ID. It also calculates the error in the colour.
    Additionally, retrieves the Gaia parallax, magnitude and effective temperature.
    
    Input
    -----
    TIC_ICs : int or list of int
        IDs from TIC_v8 catalog.
    DR : int
        Choose between DR3 (DR=3) or DR2 (DR=2)
        
    Output
    ------
    df : pandas dataframe
        Dataframe with the following columns: Gaia{}_id, tic_id, Gaia{}_G, bp_rp, Gaia{}_T
        The number 2 or 3 will be appended in the parenthesis if used DR2 or DR3
    
    """
    
    # Convert int to list
    if isinstance(TIC_IDs, (int, str)):
        TIC_IDs = [TIC_IDs]
    
    tic_data = Catalogs.query_criteria(catalog='TIC', ID=TIC_IDs) # fetch Gaia ID using TICs
    
    # mapping = {str(row['GAIA']): row['ID'] for row in tic_data if row['GAIA' != '']}
    mapping = {
        str(row['GAIA']): row['ID'] 
        for row in tic_data 
        if str(row['GAIA']).strip() not in ('', '--', 'None', 'nan')
    }
    gaia_dr2_ids = list(mapping.keys())
    
    if not gaia_dr2_ids:
        print("No valid Gaia IDs found for provided TICs.")
        return None
    
    failed_tics = [
        row['ID'] for row in tic_data 
        if str(row['GAIA']).strip() == '--'
    ]

    if failed_tics: print(f"Skipping {len(failed_tics)} stars because Gaia ID is '--'")
    
    id_list_str = "(" + ",".join(gaia_dr2_ids) + ")"
    
    if DR == 3: # Query DR3
        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
        
        query = f"""
        SELECT 
            dr3.source_id AS Gaia3_id,
            bridge.dr2_source_id AS dr2_id,
            bridge.angular_distance,
            dr3.parallax AS Gaia3_parallax,
            dr3.phot_g_mean_mag AS Gaia3_G,
            dr3.teff_gspphot AS Gaia3_T,
            dr3.bp_rp,
            dr3.phot_bp_mean_flux_over_error AS bp_err,
            dr3.phot_rp_mean_flux_over_error AS rp_err,
            dr3.ebpminrp_gspphot AS ext
        FROM gaiadr3.gaia_source AS dr3
        INNER JOIN gaiaedr3.dr2_neighbourhood AS bridge 
            ON dr3.source_id = bridge.dr3_source_id
        WHERE bridge.dr2_source_id IN {id_list_str}
        """
        
    elif DR == 2: # Query DR2
        Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"
        
        query = f""" 
        SELECT
            source_id AS Gaia2_id,
            parallax AS Gaia2_parallax,
            phot_g_mean_mag AS Gaia2_G,
            teff_val AS Gaia2_T,
            bp_rp,
            phot_bp_mean_flux_over_error AS bp_err,
            phot_rp_mean_flux_over_error AS rp_err
        FROM gaiadr2.gaia_source
        WHERE source_id IN {id_list_str}
        """
    
    else:
        raise ValueError("Invalid DR. Choose 2 or 3 to query either DR2 or DR3.")
    
    job = Gaia.launch_job(query)
    table = job.get_results()
    
    # Sort for duplicates for DR3
    if DR == 3:
        table.sort(['dr2_id', 'angular_distance'])
        _, indices = np.unique(table['dr2_id'], return_index=True)
        table = table[indices]
    
    # Error calculations
    table['bp_err'] = (2.5 / np.log(10)) / table['bp_err'] 
    table['rp_err'] = (2.5 / np.log(10)) / table['rp_err'] 
    table['bp_rp_err'] = np.sqrt(table['bp_err']**2 + table['rp_err']**2)
    table.remove_columns(['bp_err', 'rp_err'])
    
    if DR == 3: table.remove_columns(['angular_distance'])
    
    df = table.to_pandas()
    df['TIC_id'] = df['dr2_id'].astype(str).map(mapping)
    cols = df.columns.tolist()
    cols.insert(1, cols.pop(cols.index('TIC_id')))
    df = df[cols]
    
    df.insert(3, f'Gaia{DR}_dist', 1 / (1e-3 * df[f'Gaia{DR}_parallax'])) # Parllax is in mas, convert to as 
    if DR == 3: df = df.drop(columns='dr2_id').reset_index(drop=True)
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
        Dataframe for the target system(s).
    
    method : str
        What method to use, either 'B-V' (DR3) or 'BP-RP' (DR2)
    
    filepath : str
        Path to database
        
    Output
    ------
    results_dic : dictionary
        Subgrid of simulations with closest primary star to target system(s). 
    """
    
    conn = sqlite3.connect(filepath)
    results_dict = {}
    
    col_name = 'b_v' if method.lower() == 'b-v' else 'bp_rp'
    sql_col = 'Primary B-V' if method.lower() == 'b-v' else 'Primary BP-RP'
    
    for _, row in df_target.iterrows():
        target_id = row['TIC_id']
        colour_val = row[col_name]
        
        query = f"""
        SELECT * FROM "F5-K4"
        WHERE ABS("{sql_col}" - ?) = (SELECT MIN(ABS("{sql_col}" - ?)) FROM "F5-K4")
        """
        
        df_grid = pd.read_sql(query, conn, params=(colour_val, colour_val))
        df_grid = df_grid[df_grid['Primary T'] >= df_grid['Secondary T']].copy()
        results_dict[target_id] = df_grid
        
    conn.close()
    return results_dict

def querySecondary(df_grid, depth_diff: float, width_diff: float, depth_format='percentage', width_format='percentage', identical_depth=0.5, primary_width=0.0):
    """
    Query the primary subgrid to determine the most likely binary configuration, using eclipse depths and widths.
    
    Input
    -----
    **df_grid : Pandas Dataframe**
        Contains the primary subgrid to query to.
        
    **depth_diff : float**
        Depth difference between the primary and secondary eclipse. 
        Can be given in two different formats, given the input specified:
            1. Percentage - Percentage difference between the primary & secondary depth from the identical case (normalised depth of 0.5).
            2. Actual - Actual difference between the primary & secondary depth. Can supply what the depth would be for the identical case (default is 0.5, assuming normalised depth input).
            
    **width_diff : float**
        Width difference between the primary and secondary eclipse.
        Can be given in two different formats, given the input specified:
            1. Percentage - Percentage difference between the primary & secondary width, from the primary width.
            2. Actual - Actual difference between the primary & secondary width. **Must supply the primary width as primary_width**.
            
    **depth_format : str**
        The input format of the depths. Can choose between two options:
        1. Percentage - Depths are scaled with respect to an ideal case (depth =0.5).
        2. Absolute - Depths are absolute values.
        
    **width_format : str**
        The input format of the widths. Can choose between two options:
        1. Percentage - Widths are scaled already with respect to the primary widths.
        2. Absolute - Widths are absolute values.
        
    **identical_depth : float** (used if input_format=Actual)
        What value the depth would be if the binary system was identical. Defaults to 0.5, assuming the inputted depths are normalised.
        
    **primary_width : float** (used if input_format=Actual)
        What value the primary width is. Must be supplied.
        
    Output
    ------
    **df_config : Pandas dataframe**
        The best fit binary configuration, containing information about the components temperatures and radii.
    """
    
    # Handle input
    options = {'percentage', 'absolute'}
    if depth_format.lower() not in options or width_format.lower() not in options: raise ValueError('Incorrect depth or width format.')
    if depth_format.lower() == 'absolute': depth_diff *= 100 / identical_depth
    if width_format.lower() == 'absolute': width_diff *= 100 / primary_width
    
    # Filter dataframe, give depths and widths equal weighting
    df_internal = df_grid.copy()
    df_internal['depth_t1'] = df_internal['Depth Difference'] - depth_diff
    df_internal['width_t1'] = df_internal['Width Difference'] - width_diff
    
    # 2. Prevent Division by Zero
    depth_std = df_internal['Depth Difference'].std()
    width_std = df_internal['Width Difference'].std()
    
    # Use 1.0 as a fallback if std is 0 or NaN (prevents breaking the math)
    depth_std = depth_std if (depth_std > 0) else 1.0
    width_std = width_std if (width_std > 0) else 1.0

    df_internal['depth_t2'] = df_internal['depth_t1'] / depth_std
    df_internal['width_t2'] = df_internal['width_t1'] / width_std
    
    # 3. Calculate distance
    df_internal['total_diff'] = np.sqrt(df_internal['depth_t2']**2 + df_internal['width_t2']**2)
    
    # Find index, ensuring we handle potential NaNs safely
    best_idx = df_internal['total_diff'].idxmin()
    
    if pd.isna(best_idx):
        return pd.DataFrame()

    return df_grid.loc[[best_idx]]

def calculateMagnitude(df):
    df_internal = df.copy()
    