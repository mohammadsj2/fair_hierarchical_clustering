# from helper_functions_gen import find_maximal_clusters, calculate_balance_clusters, calculate_hc_obj, condense_dist, average_linkage, convert_dist, get_mw_upper_bound, get_cossim
# from eps_local_opt_fairlet import load_data_with_color, subsample, calculate_distance, find_eps_local_opt, find_eps_local_opt_random, random_fairlet_decompose, calculate_obj
# import helper_functions_gen
from eps_local_opt_fairlet import load_data_with_color
from helper_functions_gen import subsample, calculate_distance, convert_dist, average_linkage, maximum_linkage, refine_rebalance, print_tree, get_nodes, get_leaves, get_all_clusters
from helper_functions_gen import fair_hc, find_maximal_clusters, calculate_balance_clusters, calculate_hc_obj, get_mw_upper_bound, rebalance_tree, calculate_cost_obj, calculate_revenue_obj
from copy import deepcopy

import random
import math
import numpy as np
import time
import os
import sys

from datetime import datetime
def test(filename, num_list, b, r, output_direc, num_instances, c_list, ks_list, delta_list):
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
        for delta in delta_list:
            ratios = []
            for c in c_list:
                
                for ks in ks_list:
                    
                    for i in range(num_instances):
                        print("Delta: %f" %delta)
                        time_f.write("{} ".format(delta))
                        obj_f.write("{}\n".format(delta))
                        balance_f.write("{} ".format(delta))
                        ratio_f.write("{} ".format(delta))

                        print("c: %d" %c)
                        time_f.write("{} ".format(c))
                        obj_f.write("{}\n".format(c))
                        balance_f.write("{} ".format(c))
                        ratio_f.write("{} ".format(c))
                        
                        print("k: %d" %ks)
                        time_f.write("{} ".format(ks))
                        obj_f.write("{}\n".format(ks))
                        balance_f.write("{} ".format(ks))
                        ratio_f.write("{} ".format(ks))

                        print("instance: %f" %i)
                        time_f.write("{} ".format(i))
                        obj_f.write("{}\n".format(i))
                        balance_f.write("{} ".format(i))
                        ratio_f.write("{} ".format(i))

                        try:
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
                            avlk_rev_obj = calculate_revenue_obj(simi, avlk_root)
                            print("Average Linkage Cost:")
                            print(avlk_obj, avlk_rev_obj)
                            

                            bal6_tree = rebalance_tree(avlk_dummy)

                            #bal6_clusters = find_maximal_clusters(bal6_tree, b+r)

                            n = len(get_leaves(bal6_tree))
                            eps = 1 / (c * math.log2(n))
                            balance_tree = refine_rebalance(deepcopy(bal6_tree), eps)

                            h = round(n ** delta)
                            fair_tree    = fair_hc(deepcopy(balance_tree), eps, math.log2(h), ks, len(blue_pts_sample))

                            fair_obj = calculate_cost_obj(simi, fair_tree)
                            fair_rev_obj = calculate_revenue_obj(simi, fair_tree)

                            print("FairHC Cost:")
                            print(fair_obj,fair_rev_obj)
                            ratio_1 = float(fair_obj / avlk_obj)
                            ratio_2 = float(fair_rev_obj/ avlk_rev_obj)
                            ratios = ratios + [ratio_1]
                            print("Cost Ratio:")
                            print(ratio_1, ratio_2)

                            upper_bound = get_mw_upper_bound(simi)
                            print(" ================================ ")

                            obj_f.write("{} {} {} {} {} {} {}\n".format(avlk_obj, fair_obj, ratio_1, upper_bound, avlk_rev_obj, fair_rev_obj, ratio_2))



                            #################################
                            mxlk_root, _ = maximum_linkage(simi)
                            mxlk_dummy = deepcopy(mxlk_root)
                            mxlk_obj = calculate_cost_obj(simi, mxlk_root)
                            mxlk_rev_obj = calculate_revenue_obj(simi, mxlk_root)
                            
                            print("Maximum Linkage Cost:")
                            print(mxlk_obj, mxlk_rev_obj)
                            

                            bal6_tree = rebalance_tree(mxlk_dummy)

                            #bal6_clusters = find_maximal_clusters(bal6_tree, b+r)

                            n = len(get_leaves(bal6_tree))
                            eps = 1 / (c * math.log2(n))
                            balance_tree = refine_rebalance(deepcopy(bal6_tree), eps)

                            h = round(n ** delta)
                            fair_tree    = fair_hc(deepcopy(balance_tree), eps, math.log2(h), ks, len(blue_pts_sample))

                            fair_obj = calculate_cost_obj(simi, fair_tree)
                            fair_rev_obj = calculate_revenue_obj(simi, fair_tree)

                            print("FairHC Cost:")
                            print(fair_obj, fair_rev_obj)
                            ratio_1 = float(fair_obj / mxlk_obj)
                            ratio_2 = float(fair_rev_obj/ mxlk_rev_obj)
                            ratios = ratios + [ratio_1]
                            print("Cost Ratio:")
                            print(ratio_1, ratio_2)

                            upper_bound = get_mw_upper_bound(simi)
                            print(" ================================ ")

                            obj_f.write("{} {} {} {} {} {} {}\n".format(mxlk_obj, fair_obj, ratio_1, upper_bound, mxlk_rev_obj, fair_rev_obj, ratio_2))



                            obj_f.flush()
                            time_f.flush()
                            balance_f.flush()
                            ratio_f.flush()
                        except Exception as e:
                            print(e)

                    print(ratios)
                    time_f.write("\n")
                    balance_f.write("\n")
                    ratio_f.write("\n")

                time_f.write("\n")
                balance_f.write("\n")
                ratio_f.write("\n")

            time_f.write("\n")
            balance_f.write("\n")
            ratio_f.write("\n")

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
    c_list = [1, 2, 4, 8]
    ks_list = [2, 4, 8, 16] #[2, 4]
    delta_list = [1/6, 2/6, 3/6, 4/6, 5/6]
    num_instances = 5
    # c_list = [2]
    # ks_list = [2] #[2, 4]
    # delta_list = [4/6]
    # num_instances = 5
    num_list = [128] #[100, 200, 400, 800, 1600]
    np.random.seed(0)
    random.seed(0)
    output_direc = "./Results/experiments_springer"

    test(filename, num_list, b, r, output_direc, num_instances, c_list, ks_list, delta_list)


