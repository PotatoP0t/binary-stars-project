import find_period
import numpy as np
# import lightkurve as lk
# import matplotlib.pyplot as plt

#search_result = lk.search_lightcurve("TIC 381948745", author="SPOC")[-1]
# print(search_result)
# lc = search_result.download()
# time = lc.time.value
# flux = lc.flux.value

# plt.plot(time, flux)  
# plt.show()

# How close the largest eclipse has to be to 0.5 to
# be considered fully eclipsing
fully_eclipsed_tol = 0.2
# How close the eclipse depths have to be to be
# considered near-identical
identical_EB_tol = 0.1

general_file = open("EBs Data.txt", "a")
targets_file = open("Target EBs Data.txt", "a")

catalogue = open("Scripts/Jayden/hlsp_tess-ebs_tess_lcf-ffi_s0001-s0026_tess_v1.0_cat.csv", "r")
binary_stars = catalogue.readlines()
catalogue.close()
targets = []

start = 2906
end = 3135
# skipped_header = False
# Start at j = -1 to also skip the header before index 0
for j, binary_star in enumerate(binary_stars):

    # Skip the first line (header) or until starting index
    if j <= start or skipped_header == False:
        skipped_header = True
        continue

    target_id = binary_star.split(",")[0]
    targets.append(f"TIC {target_id}")
    
    if j == end: break

print(f"Index Start: {start}")
print(f"Index End: {end}")
print(f"Number of stars: {len(targets)}")

index = start
for target in targets:

    try:
        P, depths, eclipse_params, errors, chi_squ = find_period.main(target, author="SPOC")
    except Exception as e:
        print(f"ERROR in target {target}: {e}")
        index += 1
        continue

    sigma_P = errors[0]
    beta_1, beta_2, D_1, D_2, phi_1, phi_2, bl = eclipse_params
    sigma_D1 = errors[1]
    sigma_D2 = errors[2]
    A_1 = depths[0]
    sigma_A1 = errors[3]
    A_2 = depths[1]
    sigma_A2 = errors[4]

    general_file.write(f"\n{index},{target},{P},{sigma_P},{D_1},{sigma_D1},{D_2},{sigma_D2},{A_1},{sigma_A1},{A_2},{sigma_A2}")
    general_file.flush()
    # If fully eclipsing and near identical, save to targets file
    if np.abs(np.max([A_1, A_2]) - 0.5) < fully_eclipsed_tol and np.abs(A_1/A_2 - 1) < identical_EB_tol:
        targets_file.write(f"\n{index},{target},{P},{sigma_P},{D_1},{sigma_D1},{D_2},{sigma_D2},{A_1},{sigma_A1},{A_2},{sigma_A2}")
        targets_file.flush()
    
    index += 1

general_file.close()
targets_file.close()