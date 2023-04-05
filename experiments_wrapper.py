from helper_functions_v2 import calculate_distance, convert_dist, average_linkage, update_colors, RefineRebalanceTree, update_counts
from helper_functions_v2 import get_balances_at, order_children, RebalanceTree, FairHC, get_balances, print_tree, tree_cost
from eps_local_opt_fairlet import load_data_with_color
from helper_functions_gen import subsample
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import os.path
import math
import time
import random

n = 1024
save_path = "/Users/maxspringer/My Drive/Hajiaghayi Research/FairCluster/revision_figures"

# LOAD DATA INTO NUMPY ARRAY
filename = "adult.csv"
data_bal = 6/7
blue_points, red_points = load_data_with_color(filename)
blue_pts_sample, red_pts_sample = subsample(blue_points, red_points, n)
data = []
data.extend(blue_pts_sample)
data.extend(red_pts_sample)
data = np.array(data)
# Note: node ids correspond to index in data list

num_blue = len(blue_pts_sample)
num_red  = len(red_pts_sample)
blue_ids = np.arange(num_blue)
red_ids  = np.arange(num_blue, num_blue + num_red)

# BUILD AVERAGE LINKAGE TREE
dist, _ = calculate_distance(data)
simi = convert_dist(dist)

lkg_start = time.time()
root, _ = average_linkage(simi)
lkg_end   = time.time() - lkg_start
update_colors(root, red_ids, blue_ids) # Initialize colors
avg_linkage = deepcopy(root)
print(" --- Average linkage tree built! --- ")

filename = os.path.join(save_path, "cost_experiment_output.txt")
text_file = open(filename, "w")

# COLLECT CLUSTER STATISTICS OF AVG LINK
# pre_balance = np.sort(get_balances(root))
# fig, axs = plt.subplots(figsize=(4,4))
# axs.hist(pre_balance, bins=10, density=1)
# # axs.hist(pre_balance, density=1)
# axs.axvline(x=data_bal, color='r')
# axs.set_xlim([0,2*(data_bal)])
# plt.savefig(os.path.join(save_path, "avg_link_clusters.png"))

print(" Time taken was %s seconds" % lkg_end)
avg_lkg_cost = tree_cost(root,simi)
# avg_lkg_cost = 1
text_file.write(" --- Cost of average linkage tree = %s --- \n" % avg_lkg_cost)
print(" --- Running fair hierarchical clustering for various parameters... ---")

# ============================================================================================================= #
# RUN FAIR CLUSTERING ALGORITHMS

cs = [1,2,4,8] #[1,2,4,8,16]
iis = [2,3,4] #,5,6]
jjs = [1] # [1,2,3]# [1]
for c in cs:
    eps = 1/(c * math.log2(n))
    for ii in iis:
        for jj in jjs:
            print("Folding with i = ", ii, " and j = ", jj, "c = ", c)
            base_root = deepcopy(root)
            # Start the timer
            start_time = time.time()
            order_children(base_root)
            RebalanceTree(base_root)

            order_children(base_root)
            RefineRebalanceTree(base_root, eps)

            update_colors(base_root, red_ids, blue_ids)

            h = 2 ** ii
            k = 2 ** jj
            FairHC(base_root, h, k, red_ids, blue_ids)
            update_counts(root)
            update_colors(root, red_ids, blue_ids)

            end_time = time.time() - start_time
            print(" --- Finished algorithm with parameters (c,h,k) = (%d,%d,%d) in %s seconds --- \n" % (c,h,k,end_time))

            if k > 1:
                post_balance = np.sort(get_balances_at(base_root, math.log2(k)))
            else:
                post_balance = np.sort(get_balances(base_root))

            # fig, axs = plt.subplots(figsize=(4,4))
            # axs.hist(post_balance, bins=5, density=1)
            # # axs.hist(post_balance, density=1)
            # axs.axvline(x=data_bal, color='r')
            # axs.set_xlim([0,2*data_bal])
            # save_name = "post_" + "c=" + str(c) + "_h=" + str(h) + "_k=" + str(k) + ".png"
            # plt.savefig(os.path.join(save_path, save_name))

            fair_cost = tree_cost(base_root, simi)
            rel_cost = fair_cost / avg_lkg_cost
            if fair_cost == 0 or rel_cost == 0:
                print("COST IS ZERO WHY")

            text_file.write(" --- Raw Cost of FairHC tree = %s --- \n" % fair_cost)
            text_file.write(" --- Relative Cost of FairHC tree = %s --- \n" % rel_cost)

text_file.close()
print("Finished run.")