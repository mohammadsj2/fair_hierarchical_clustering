# from helper_functions_gen import find_maximal_clusters, calculate_balance_clusters, calculate_hc_obj, condense_dist, average_linkage, convert_dist, get_mw_upper_bound, get_cossim
# from eps_local_opt_fairlet import load_data_with_color, subsample, calculate_distance, find_eps_local_opt, find_eps_local_opt_random, random_fairlet_decompose, calculate_obj
# import helper_functions_gen
from eps_local_opt_fairlet import load_data_with_color
from helper_functions_gen import subsample, calculate_distance, convert_dist, average_linkage, refine_rebalance, print_tree, get_nodes, get_leaves, get_all_clusters
from helper_functions_gen import fair_hc, find_maximal_clusters, calculate_balance_clusters, calculate_hc_obj, get_mw_upper_bound, rebalance_tree, calculate_cost_obj
from copy import deepcopy

import random
import math
import numpy as np
import time
import os
import sys

from datetime import datetime
def test(filename, num_list, b, r, output_direc, num_instances, c, ks, delta):
    blue_points, red_points = load_data_with_color(filename)
    time_file = "{}/{}_{}_{}_{}.out".format(output_direc, filename.split(".")[0], "time", "b="+str(b), "r="+str(r))
    balance_file = "{}/{}_{}_{}_{}.out".format(output_direc, filename.split(".")[0],"balance","b="+str(b), "r="+str(r))
    obj_file = "{}/{}_{}_{}_{}.out".format(output_direc, filename.split(".")[0],"obj","b="+str(b), "r="+str(r))
    ratio_file = "{}/{}_{}_{}_{}.out".format(output_direc, filename.split(".")[0],"ratio","b="+str(b), "r="+str(r))
    time_f = open(time_file, "w")
    balance_f = open(balance_file, "w")
    obj_f = open(obj_file, "w")
    ratio_f = open(ratio_file, "w")

    obj_f.write("avlk_obj  fair_obj  fair_ratio  upper bound\n")
    num = num_list[0]
    for num in num_list:
        print("sample number: %d" %num)
        time_f.write("{} ".format(num))
        obj_f.write("{}\n".format(num))
        balance_f.write("{} ".format(num))
        ratio_f.write("{} ".format(num))
        ratios = []

        for i in range(num_instances):
            blue_pts_sample, red_pts_sample = subsample(blue_points, red_points, num)

            data = []
            data.extend(blue_pts_sample)
            data.extend(red_pts_sample)
            data = np.array(data)
            # first calculate the pairwise distance for all points
            start = time.time()
            dist, d_max = calculate_distance(data, dist_type="euclidean")
            simi = convert_dist(dist)

            avlk_root, _ = average_linkage(simi)
            avlk_dummy = deepcopy(avlk_root)

            avlk_obj = calculate_cost_obj(simi, avlk_root)
            print("Average Linkage Cost:")
            print(avlk_obj)

            bal6_tree = rebalance_tree(avlk_dummy)

            #bal6_clusters = find_maximal_clusters(bal6_tree, b+r)

            n = len(get_leaves(bal6_tree))
            eps = 1 / (c * math.log2(n))
            balance_tree = refine_rebalance(deepcopy(bal6_tree), eps)

            h = round(n ** delta)
            fair_tree    = fair_hc(deepcopy(balance_tree), eps, math.log2(h), k, len(blue_pts_sample))

            fair_obj = calculate_cost_obj(simi, fair_tree)
            print("FairHC Cost:")
            print(fair_obj)
            ratio_1 = float(fair_obj / avlk_obj)
            ratios = ratios + [ratio_1]
            print("Cost Ratio:")
            print(ratio_1)

            upper_bound = get_mw_upper_bound(simi)
            print(" ================================ ")

            obj_f.write("{} {} {} {} \n".format(avlk_obj, fair_obj, ratio_1, upper_bound))

            obj_f.flush()
            time_f.flush()
            balance_f.flush()
            ratio_f.flush()

        print(ratios)
        time_f.write("\n")
        balance_f.write("\n")
        ratio_f.write("\n")

    time_f.close()
    balance_f.close()
    obj_f.close()
    ratio_f.close()
    return



if __name__ == "__main__":
    sys.setrecursionlimit(100000)
    filename = "Census Race 1:7 adult_r.csv"
    b = 1
    r = 7
    c = 4 #[1, 2, 4, 8, 16]
    ks = [2, 4, 8, 16] #[2, 4]
    delta = 5/7 #[3/7, 4/7, 5/7, 6/7]
    num_instances = 10
    num_list = [128] #[100, 200, 400, 800, 1600]
    np.random.seed(0)
    random.seed(0)
    output_direc = "./Results/experiments_springer"
    test(filename, num_list, b, r, output_direc, num_instances, c, ks, delta)


