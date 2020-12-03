import numpy as np
import pandas as pd

def group_contious(arr):
    arr.sort()
    i = 0
    rtn_arr = []
    while i < len(arr):
        start_i = arr[i]
        end_i = arr[i]
        while i < len(arr)-1 and arr[i] == arr[i+1]-1:
            end_i = arr[i+1]
            i += 1
        rtn_arr.append((start_i,end_i))
        i += 1
    return rtn_arr

def double_peek_grouping(point_stack, bins=10):

    df = pd.DataFrame({'rho':point_stack[:,0], 'theta':point_stack[:,1]})

    hist_rho, edges_rho = np.histogram(df['rho'].tolist(), bins=bins)
    cumsum_rho = np.cumsum(hist_rho)
    hist_theta, edges_theta = np.histogram(df['theta'].tolist(), bins=bins)
    cumsum_theta = np.cumsum(hist_theta)

    rho_peeks = []
    theta_peeks = []

    if hist_rho[0] != 0 and hist_rho[0] >= hist_rho[1]:
        rho_peeks.append(0)

    if hist_rho[-1] != 0 and hist_rho[-1] >= hist_rho[-2]:
        rho_peeks.append(len(hist_rho)-1)

    if hist_theta[0] != 0 and hist_theta[0] >= hist_theta[1]:
        theta_peeks.append(0)

    if hist_theta[-1] != 0 and hist_theta[-1] >= hist_theta[-2]:
        theta_peeks.append(len(hist_theta)-1)

    for i in range(1, len(hist_rho)-1):
        if hist_rho[i] != 0 and hist_rho[i] >= hist_rho[i-1] and hist_rho[i] >= hist_rho[i+1]:
            rho_peeks.append(i)
        if hist_theta[i] != 0 and hist_theta[i] >= hist_theta[i-1] and hist_theta[i] >= hist_theta[i+1]:
            theta_peeks.append(i)

    rho_peeks = group_contious(rho_peeks)
    theta_peeks = group_contious(theta_peeks)

    cleaned_point_stack = []

    for rho_split in rho_peeks:
        rho_min, rho_max = edges_rho[rho_split[0]], edges_rho[rho_split[1]+1]
        matched_rho = df[df['rho'].between(rho_min, rho_max)]

        for theta_split in theta_peeks:
            theta_min, theta_max = edges_theta[theta_split[0]], edges_theta[theta_split[1]+1]

            matched_rho_theta =  df[df['theta'].between(theta_min, theta_max)]

            if matched_rho_theta.size != 0:
                sub_df_mean = matched_rho_theta.mean()
                cleaned_point_stack.append([sub_df_mean['rho'], sub_df_mean['theta']])
    return cleaned_point_stack


# def average_line(img, line_info):
#     h, w = img.shape[:2]
#
#     # Compute Average Lines
#     total_m = 0
#     total_b = 0
#     for coeff in pos_lines:
#         total_m += coeff[0]
#         total_b += coeff[1]
#
#     pos_m = total_m / len(pos_lines)
#     pos_b = total_b / len(pos_lines)
#
#     total_m = 0
#     total_b = 0
#     for coeff in neg_lines:
#         total_m += coeff[0]
#         total_b += coeff[1]
#
#     neg_m = total_m / len(neg_lines)
#     neg_b = total_b / len(neg_lines)
#
#     y1 = pos_m * 100 + pos_b
#     y2 = pos_m * (w - 100) + pos_b
#
#     y1 = neg_m * 100 + neg_b
#     y2 = neg_m * (w - 100) + neg_b
#
#     return
