#this file is the helper file for the average linkage tracking
from eps_local_opt_fairlet import load_data_with_color
from helper_functions_gen import subsample
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import math
import time
import random

def calculate_distance(points, dist_type = "euclidean"):
    n = len(points)
    dist = np.zeros((n,n))
    d_max = 0.0

    for i in range(1,n):
        for j in range(i):
            dif = np.array(points[i]) - np.array(points[j])
            d = np.sqrt(np.einsum("i,i->", dif, dif))
            dist[i][j] = d
            dist[j][i] = d
            if d > d_max:
                d_max = d
    return dist, d_max

#simi = 1/(1+d)
def convert_dist(dist):
    n = dist.shape[0]
    simi = np.copy(dist)
    for i in range(n):
        for j in range(n):
            if i != j:
                simi[i][j] = 1 / (simi[i][j] + 1)
    return simi

#return s_max with u<v (indices)
def find_max(simi, x):
    u = 0
    v = 0
    d_max = -math.inf
    for i in x:
        for j in x:
            if i == j:
                continue
            if simi[i][j] > d_max:
                u = np.minimum(i,j)
                v = np.maximum(i,j)
                d_max = simi[i][j]
    return u,v

def update_simi(simi, left_index, right_index, left_weight, right_weight):
    new_row = (simi[left_index,:] * left_weight + simi[right_index,:] * right_weight) / (left_weight + right_weight)
    simi = np.vstack((simi, new_row))
    new_row = np.append(new_row, 0)
    new_column = new_row.reshape((-1,1))
    simi = np.hstack((simi, new_column))
    simi = np.delete(simi, [left_index, right_index], axis=0)
    simi = np.delete(simi, [left_index, right_index], axis=1)
    return simi

class Node:
    def __init__(self, id = None, parent = None, children = None, count = 0, color = None):
        self.id = id
        self.parent = parent
        self.children = children
        self.count = count
        self.color = color # want color feature to reflect number of red points / count

    def get_count(self):
        return self.count

    def get_id(self):
        return self.id

    def get_children(self):
        return self.children

    def get_parent(self):
        return self.parent

    def get_color(self):
    	return self.color

    def is_leaf(self):
        if self.children == None:
            return True

def print_tree(node, s=""):
    print(s, node.get_id(), node.get_count())
    if not node.is_leaf():
        for child in node.get_children():
            print_tree(child, "\t" + s[:-3] + "|--")

def print_tree_color(node, s=""):
    print(s, node.get_id(), node.get_count(), round(node.get_color(),2))
    if not node.is_leaf():
        for child in node.get_children():
            print_tree_color(child, "\t" + s[:-3] + "|--")

def average_linkage(simi, current_id=None, indices=None, leaves=None):
    n = simi.shape[0]
    if current_id is None:
        current_id = n
    if leaves is None:
        if indices is not None:
            leaves = [Node(id=id, parent=None, children=None, count=1) for id in indices]
        else:
            leaves = [Node(id=id, parent=None, children=None, count=1) for id in range(n)]

    while len(leaves) > 1:
        left_index, right_index = find_max(simi, range(len(leaves)))

        left_node = leaves[left_index]
        right_node = leaves[right_index]
        left_weight = left_node.get_count()
        right_weight = right_node.get_count()
        new_node = Node(id=current_id, parent=None, children=[left_node, right_node],count=left_weight + right_weight)
        left_node.parent = new_node
        right_node.parent = new_node
        current_id += 1
        leaves.append(new_node)
        simi = update_simi(simi, left_index, right_index, left_weight, right_weight)
        del leaves[right_index]
        del leaves[left_index]

    return leaves[0], current_id

def update_colors(root, red_ids, blue_ids):
    '''
    Takes as input a tree and red / blue node ids. Updates the color 
    value for each node to be num. red points / count (1 - this value is blue fraction)
    '''
    if root.is_leaf():
        if root.get_id() in red_ids:
            root.color = 1
        else:
            root.color = 0
        return

    color_counter = 0
    for child in root.get_children():
        update_colors(child, red_ids, blue_ids)
        color_counter += child.get_color() * child.get_count()
    root.color = color_counter / root.get_count()
    return

## ------------------ ALGORITHMS ------------------ ##

def RebalanceTree(root):
    if root.is_leaf():
        return
    v = root
    n = root.get_count()
    A = (v.get_children()[0]).get_count()
    while A >= (2/3) * n:
        v = v.get_children()[0]
        A = (v.get_children()[0]).get_count()
    v = v.get_children()[0]
    tree_rebalance(v,root)
    RebalanceTree(root.get_children()[0])
    RebalanceTree(root.get_children()[1])

def RefineRebalanceTree(root, eps):
    n = root.get_count()
    if eps <= 1/(2*n):
        return

    v = root
    Tbig   = v.get_children()[0]
    Tsmall = v.get_children()[1]

    while Tbig.get_count() >= (1/2 + eps) * n:
        delta = (Tbig.get_count() - (n / 2))/n
        SubtreeSearch(v, delta * n)
        Tbig   = v.get_children()[0]
        Tsmall = v.get_children()[1]
    Tbig = RefineRebalanceTree(Tbig, eps)
    Tsmall = RefineRebalanceTree(Tsmall, eps)
    update_counts(root)
    order_children(root)

def SubtreeSearch(root, s):
    '''
    Takes as input 1/6-balanced tree with ordered children by size and error parameter s
    '''
    v = root
    while (v.get_children()[1]).get_count() > s:
        v = v.get_children()[0]
    v = v.get_children()[1]

    u = root
    while not u.is_leaf() and (u.get_children()[0]).get_count() >= v.get_count():
    # while (u.get_children()[0]).get_count() >= v.get_count():
        u = u.get_children()[1]
    del_ins(root, u, v)
    patch_compression(root)
    # return root

def FairHC(root, h, k, red_ids, blue_ids):
    '''
    Input should be eps = 1/(c * lg n) rebalance
    h = 2^i
    k = 2^j
    0 < j < i
    '''
    if root.is_leaf() or is_final(root):
        return
    level_abstract(root, 0, math.log2(h))
    update_counts(root)
    update_colors(root, red_ids, blue_ids)
    if is_final(root): # if height of tree is 1, return tree
    # if root.is_leaf():
        return

    for c in range(2):
        for i in range(k):
            order_children(root, c+1) # order by color
            to_be_folded = []
            for j in range(round(h/k)-1):
                to_be_folded.append(root.get_children()[i + j*k])
            fold(root, to_be_folded)
            update_counts(root)
            update_colors(root, red_ids, blue_ids)

    for child in root.get_children():
        FairHC(child, h, k, red_ids, blue_ids)
    return

## ---------------- TREE OPERATORS ---------------- ##

def tree_rebalance(u, v):
    '''
    Rebalance u at node v
    '''
    # Base case
    if u.get_parent().get_id() == v.get_id() or u.get_id() == v.get_id() or u.is_leaf():
        return
    root_children = v.get_children()
    children_counts = v.get_count()

    u_parent  = u.get_parent()
    for child in u_parent.get_children():
        if child.get_id() != u.get_id():
            u_sibling = child

    # Remove u
    u_parent.children = [u_sibling] # remove u from u_parent's children

    # Check if need to contract u's sibling
    if not u_sibling.is_leaf():
        u_grand = u_parent.get_parent()
        u_sibling.parent = u_grand
        if u_grand is not None: # if u's parent is NOT the new tree root
            u_grand.children.remove(u_parent)
            u_grand.children.append(u_sibling)

    unique_id = get_max_id(v) + 1

    if v.get_count() - u.get_count() == 1: # v is a leaf!
        v.children.append(u)
    else:
        new_node = Node(id=unique_id, parent = v, children = v.get_children(), count = v.get_count() - u.get_count())
        v.children = [u, new_node]

        # Update parent pointers and counts
        for child in new_node.get_children():
            child.parent = new_node
    u.parent = v
    update_counts(v)
    patch_compression(v)
    order_children(v)

def del_ins(root, u, v):
    '''
    Remove v and insert it at node u
    '''

    # Collect v's parent and sibling
    v_parent = v.get_parent()
    for child in v_parent.get_children():
        if child.get_id() != v.get_id():
            v_sibling = child

    # Remove v
    v_parent.children = [v_sibling]

    # Check if need to contract v's sibling
    if not v_sibling.is_leaf():
        v_grand = v_parent.get_parent()
        v_sibling.parent = v_grand
        if v_grand is not None: # if u's parent is NOT the new tree root
            v_grand.children.remove(v_parent)
            v_grand.children.append(v_sibling)

    unique_id = get_max_id(root) + 1
    new_node = Node(id = unique_id, parent = u.get_parent(), children = [v,u], count = v.get_count() + u.get_count())

    # Update parent pointers and counts
    u.parent = new_node
    v.parent = new_node
    grand = new_node.get_parent()
    grand.children.remove(u)
    grand.children.append(new_node)

    update_counts(root)
    # order_children(root)

def level_abstract(root, h1, h2):
    # Get nodes at level h1
    level_h1_nodes = get_nodes_at_level(root, h1)
    level_h2_nodes = get_nodes_at_level(root, h2)

    for h1_node in level_h1_nodes:
        new_children = []
        for h2_node in level_h2_nodes:
            if is_descendant(h1_node, h2_node): # Implement the is_descendant function
                new_children = new_children + [h2_node]
                h2_node.parent = h1_node
        h1_node.children = new_children

def fold(root, trees):
    unique_id    = get_max_id(root) + 1
    new_node     = Node(id = unique_id, parent = trees[0].get_parent(), children = None, count = 0, color = 0)
    new_children = []
    new_count    = 0

    for tree in trees:
        new_count += tree.get_count()
        if tree.is_leaf():
            new_children.append(tree)
            tree.parent = new_node
            break
        tree.parent = None
        for child in tree.get_children():
            new_children.append(child)
            child.parent = new_node
        # tree.children = None

    new_node.children = new_children
    new_node.count = new_count
    # update_colors(root)

## --------------- HELPER FUNCTIONS --------------- ##

def check_isomorphic(t1, t2):
    if t1.is_leaf() and t2.is_leaf():
        return True
    elif t1.is_leaf() or t2.is_leaf():
        return False
    elif is_final(t1) and is_final(t2):
        return True
    elif len(t1.get_children()) != len(t2.get_children()):
        return False
    cond1 = check_isomorphic(t1.get_children()[0], t2.get_children()[0])
    cond2 = check_isomorphic(t1.get_children()[1], t2.get_children()[1])
    return cond1 and cond2

def is_final(node):
    for child in node.get_children():
        if not child.is_leaf():
            return False
    return True

def is_descendant(node1, node2):
    if node2.get_parent().get_id() == node1.get_id():
        return True
    elif node1.is_leaf():
        return False
    else:
        verdict = False
        for child in node1.get_children():
            if is_descendant(child, node2):
                verdict = True
        return verdict

def order_children(node, sort_by=0):
    '''
    Order a node's children by their key (descending)
    sort_by = 0 to sort by count, = 1 to sort by color
    '''
    if not node.is_leaf():
        if sort_by == 0: # Sort by total counts
            node.children.sort(key=lambda x: x.count, reverse=True)
        if sort_by == 1: # Sort by red counts
            node.children.sort(key=lambda x: x.color, reverse=True)
        if sort_by == 2: # Sort by blue counts
            node.children.sort(key=lambda x: 1-x.color, reverse=True)
        for child in node.get_children():
            order_children(child)

def get_node(root, id):
    '''
    Get the node of a certain id from the tree
    '''
    if root.get_id() == id:
        return root

    if not root.is_leaf():
        for child in root.get_children():
            search_result = get_node(child,id)
            if search_result is not None:
                return search_result

def list_nodes(root):
    '''
    List all nodes in a given tree
    '''
    all_nodes = []
    all_nodes.append(root)

    if root.is_leaf():
        return all_nodes

    for child in root.get_children():
        all_nodes = all_nodes + list_nodes(child)
    
    return all_nodes

def list_leaves(root):
    '''
    List all leaves in a given tree
    '''
    if root.is_leaf():
        return [root]

    all_leaves = []
    for child in root.get_children():
        all_leaves = all_leaves + list_leaves(child)
    return all_leaves

def get_max_id(root):
    '''
    Get the maximal node id in a tree
    '''
    if root.get_parent() is not None:
        return get_max_id(root.get_parent())

    all_nodes = list_nodes(root)
    max_id = -1
    for node in all_nodes:
        if node.get_id() > max_id:
            max_id = node.get_id()
    return max_id

def update_counts(root):
    '''
    Update the node counts (to be run after a tree operation)
    '''
    if root.is_leaf():
        root.count = 1
    else:
        root.count = len(list_leaves(root))
        for child in root.get_children():
            update_counts(child)

def check_balance(root, eps):
    '''
    DEBUG THIS, IT FAILS A PROPER CHECK
    '''
    if not root.is_leaf():
        n = root.get_count() # len(list_leaves(root))
    if n > 1/(2 * eps):
        for child in root.get_children():
            lb = (1/2 - eps) * n
            ub = (1/2 + eps) * n
            c_size = child.get_count() # len(list_leaves(child))
            if c_size < lb or c_size > ub:
                return False
            else:
                return check_balance(child, eps)
    return True

def patch_compression(root):
    if root.get_count() == 1 and not root.is_leaf():
        leaf_child = root.get_children()[0]
        leaf_child.parent = root.get_parent()
        root.get_parent().children.remove(root)
        root.get_parent().children.append(leaf_child)

        root.parent = None
        root.children = None
    elif root.get_count() > 1:
        for child in root.get_children():
            patch_compression(child)

def get_nodes_at_level(root, h):
    if h == 0 or root.is_leaf():
        return [root]

    collected_nodes = []
    if root.get_children() is None:
        return []
    for child in root.get_children():
        collected_nodes = collected_nodes + get_nodes_at_level(child, h-1)
    return collected_nodes

def get_balances(root):
    if root.is_leaf(): return []

    cluster_balances = []
    cluster_balances += [root.get_color()]
    for child in root.get_children():
        cluster_balances += get_balances(child)
    return cluster_balances

def get_balances_at(root, level):
    cluster_balances = []
    nodes = get_nodes_at_level(root, level)
    for node in nodes:
        cluster_balances += [node.get_color()]
    return cluster_balances

def get_lca(node1, node2):
    # Pick 1 node, is it the parent of the other?
    # keep checking is descendant until parent is found

    parent1 = node1.get_parent()
    h1      = get_node_depth(parent1)
    parent2 = node2.get_parent()
    h2      = get_node_depth(parent2)

    while h1 != h2:
        if h1 > h2:
            parent1 = parent1.get_parent()
            h1      = get_node_depth(parent1)
        else:
            parent2 = parent2.get_parent()
            h2      = get_node_depth(parent2)
    while parent1.get_id() != parent2.get_id():
        if parent1.get_parent() is None or parent2.get_parent() is None:
            return parent1
        parent1 = parent1.get_parent()
        parent2 = parent2.get_parent()
    return parent1

def tree_cost(root, simi):
    cost = 0
    num_pts = simi.shape[0]

    for x in range(num_pts):
        x_node = get_node(root, x)
        if x_node is None:
            continue
        for y in range(x, num_pts):
            y_node = get_node(root, y)
            if x == y or y_node is None:
                continue
            lca = get_lca(x_node, y_node)
            num_leaves = lca.get_count()
            cost = cost + (num_leaves * simi[x][y])
            # print(cost)
    return cost

def get_node_depth(node):
    if node.get_parent() is None:
        return 0
    else:
        return 1 + get_node_depth(node.get_parent())

def pointer_patch(root):
    if root.is_leaf():
        return

    for child in root.get_children():
        if child.get_parent() is None or child.get_parent().get_id() != root.get_id():
            child.parent = root
        pointer_patch(child)

## ------------------------------------------------ ##

if __name__ == "__main__":
    n = 128
    c = 1
    eps = 1 / (c * math.log2(n)) # 1/16
    h = 4
    k = 2

    filename = "adult.csv"
    blue_points, red_points = load_data_with_color(filename)
    blue_pts_sample, red_pts_sample = subsample(blue_points, red_points, n)
    data = []
    data.extend(blue_pts_sample)
    data.extend(red_pts_sample)
    data = np.array(data)
    # Note: node ids correspond to index in data list! (I think)
    num_blue = len(blue_pts_sample)
    num_red  = len(red_pts_sample)
    blue_ids = np.arange(num_blue)
    red_ids  = np.arange(num_blue, num_blue + num_red)

    dist, _ = calculate_distance(data)
    simi = convert_dist(dist)
    root, _ = average_linkage(simi)
    update_colors(root, red_ids, blue_ids) # Initialize colors

    print("Node 10 is at depth ", get_node_depth(get_node(root, 10)))


    # print(" --------------------------------------------------- ")
    # print("Cost of average linkage tree is ", tree_cost(root, simi))
    # print(" --------------------------------------------------- ")

    # order_children(root)
    # RebalanceTree(root)

    # order_children(root)
    # RefineRebalanceTree(root, eps)

    # # # Update colors before folding operations
    # update_colors(root, red_ids, blue_ids)
    # FairHC(root, h, k, red_ids, blue_ids)
    # update_counts(root)
    # update_colors(root, red_ids, blue_ids)
    # # print_tree(root)

    # pointer_patch(root)

    # print(" --------------------------------------------------- ")
    # print("Cost of fair tree is ", tree_cost(root,simi))
    # print(" --------------------------------------------------- ")




    


