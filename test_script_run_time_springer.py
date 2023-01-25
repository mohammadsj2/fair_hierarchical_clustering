# from helper_functions_gen import find_maximal_clusters, calculate_balance_clusters, calculate_hc_obj, condense_dist, average_linkage, convert_dist, get_mw_upper_bound, get_cossim
# from eps_local_opt_fairlet import load_data_with_color, subsample, calculate_distance, find_eps_local_opt, find_eps_local_opt_random, random_fairlet_decompose, calculate_obj
# import helper_functions_gen
from eps_local_opt_fairlet import load_data_with_color
from helper_functions_gen import subsample, calculate_distance, convert_dist, average_linkage, refine_rebalance, print_tree, get_nodes, get_leaves
from helper_functions_gen import fair_hc, find_maximal_clusters, calculate_balance_clusters, calculate_hc_obj, get_mw_upper_bound, rebalance_tree, calculate_cost_obj
from copy import deepcopy

import random
import math
import numpy as np
import time
import os
import sys

from datetime import datetime
def test(filename, num_list, b, r, output_direc, num_instances, c, k, delta):
    blue_points, red_points = load_data_with_color(filename)
    time_file = "{}/{}_{}_{}_{}.out".format(output_direc, filename.split(".")[0], "time", "b="+str(b), "r="+str(r))
    balance_file = "{}/{}_{}_{}_{}.out".format(output_direc, filename.split(".")[0],"balance","b="+str(b), "r="+str(r))
    obj_file = "{}/{}_{}_{}_{}.out".format(output_direc, filename.split(".")[0],"obj","b="+str(b), "r="+str(r))
    ratio_file = "{}/{}_{}_{}_{}.out".format(output_direc, filename.split(".")[0],"ratio","b="+str(b), "r="+str(r))
    #swap_file = "{}/{}_{}_{}_{}.out".format(output_direc, filename.split(".")[0],"swap","b="+str(b), "r="+str(r))
    #random_file = "{}/{}_{}_{}_{}.out".format(output_direc, filename.split(".")[0],"random","b="+str(b), "r="+str(r))

    time_f = open(time_file, "w")
    balance_f = open(balance_file, "w")
    obj_f = open(obj_file, "w")
    ratio_f = open(ratio_file, "w")
    #swap_f = open(swap_file, "w")
    #random_f = open(random_file, "w")

    obj_f.write("avlk_obj  fair_obj  fair_ratio  upper bound\n")

    for num in num_list:
        print("sample number: %d" %num)
        time_f.write("{} ".format(num))
        obj_f.write("{}\n".format(num))
        balance_f.write("{} ".format(num))
        ratio_f.write("{} ".format(num))
        #swap_f.write("{} ".format(num))
        #random_f.write("{} ".format(num))

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

            # fairlets, swap_counter, random_counter, random_fairlets = find_eps_local_opt_random(blue_pts_sample, red_pts_sample, dist, d_max, b, r, eps, rho)

            # end = time.time()

            # total_dist = np.sum(np.sum(dist)) / 2
            # dist_ratio = float(calculate_obj(fairlets, dist) / total_dist)

            # t = end - start
            # time_f.write("{} ".format(t))
            # ratio_f.write("{} ".format(dist_ratio))
            #swap_f.write("{} ".format(swap_counter))
            #random_f.write("{} ".format(random_counter))
            # print("total time spent on finding fairlets: %f s" % t)
            #print("number of swaps: %d" %swap_counter)
            #print("number of randomization: %d" %random_counter)

            avlk_root, _ = average_linkage(simi)
            avlk_dummy = deepcopy(avlk_root)
            # print(" ======================================= ")
            # print_tree(avlk_root)
            # print_tree(avlk_root)

            ######################
            ##### ADDED BITS #####
            ######################




            ######################
            ######################

            # find the balance of maximal clustering
            # print("Blue point count:")
            # print(len(blue_pts_sample))

            avlk_maximal_cluster = find_maximal_clusters(avlk_root, b + r)
            # avlk_root_balance = calculate_balance_clusters(avlk_maximal_cluster, len(blue_points))
            avlk_root_balance = calculate_balance_clusters(avlk_maximal_cluster, len(blue_points))
            # print("AVLK Balance:")
            # print(avlk_root_balance)
            balance_f.write("{} ".format(avlk_root_balance))
            avlk_obj = calculate_cost_obj(simi, avlk_root)
            print("Average Linkage Cost:")
            print(avlk_obj)

            bal6_tree = rebalance_tree(avlk_dummy)
            #bal6_clusters = find_maximal_clusters(bal6_tree, b+r)
            print("BAL6 Cost:")
            print(calculate_cost_obj(simi,bal6_tree))

            n = len(get_leaves(bal6_tree))
            eps = 1 / (c * math.log2(n))
            balance_tree = refine_rebalance(deepcopy(bal6_tree), eps)
            print("REFINE Cost:")
            print(calculate_cost_obj(simi,balance_tree))

            # h = n ** delta
            h = 4
            # print(h)
            # fair_tree    = fair_hc(deepcopy(balance_tree), eps, h, k, len(blue_points))
            fair_tree    = fair_hc(deepcopy(balance_tree), eps, h, k, len(blue_points))
            # print_tree(fair_tree)
            # print_tree(fair_tree)

            # find the balance of maximal clustering after rebalance
            fair_maximal = find_maximal_clusters(fair_tree, b + r)
            # fair_balance = calculate_balance_clusters(fair_maximal, len(blue_points))
            fair_balance = calculate_balance_clusters(fair_maximal, len(blue_points))
            print("FairHC Balance:")
            print(fair_balance)
            # print(fair_balance)
            balance_f.write("{} ".format(fair_balance))

            # find how much we lose by being fair
            # print_tree(avlk_root)
            # avlk_obj = calculate_hc_obj(simi, avlk_root)


            # fair_root = avlk_with_fairlets(simi, fairlets)
            # fair_obj = calculate_hc_obj(simi, fair_root)
            # print_tree(fair_tree)
            # fair_obj = calculate_hc_obj(simi, fair_tree)
            fair_obj = calculate_cost_obj(simi, fair_tree)
            print("FairHC Cost:")
            print(fair_obj)
            ratio_1 = float(fair_obj / avlk_obj)
            print("Cost Ratio:")
            print(ratio_1)

            # print_tree(avlk_root)
            # print_tree(fair_tree)

            upper_bound = get_mw_upper_bound(simi)
            print(" ================================ ")
            # 

            # random_root = avlk_with_fairlets(simi, random_fairlets)
            # random_obj = calculate_hc_obj(simi, random_root)
            # ratio_2 = float(random_obj / avlk_obj)


            obj_f.write("{} {} {} {} \n".format(avlk_obj, fair_obj, ratio_1, upper_bound))

            obj_f.flush()
            time_f.flush()
            balance_f.flush()
            ratio_f.flush()
            #swap_f.flush()
            #random_f.flush()

        time_f.write("\n")
        balance_f.write("\n")
        ratio_f.write("\n")
        #swap_f.write("\n")
        #random_f.write("\n")

    time_f.close()
    balance_f.close()
    obj_f.close()
    ratio_f.close()
    #swap_f.close()
    #random_f.close()
    return



if __name__ == "__main__":
    sys.setrecursionlimit(100000)
    filename = "adult.csv" # "adult_r.csv"
    b = 1
    r = 7
    c = 4
    k = 2
    delta = 2/7
    # eps = 1 / (c * math.log2(n))
    # h = n ** delta
    num_instances = 5
    num_list = [128]#[100, 200, 400, 800, 1600]
    np.random.seed(0)
    random.seed(0)
    # output_direc = "./experiments_springer"
    output_direc = "/Users/maxspringer/Documents/GitHub/fair_hierarchical_clustering/experiments_springer"
    test(filename, num_list, b, r, output_direc, num_instances, c, k, delta)


