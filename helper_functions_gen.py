#this file is the helper file for the average linkage tracking
from copy import deepcopy
import numpy as np
import math
import time
import random

class Node:
    def __init__(self, id = None, children = None, count = 0, color=None):
        self.id = id
        self.children = children
        self.count = count
        self.color = color

    def get_count(self):
        return self.count

    def get_id(self):
        return self.id

    def get_children(self):
        return self.children

    def get_color(self):
    	return self.color

    def is_leaf(self):
        if self.children == None:
            return True
        return False


def subsample(blue_points, red_points, num):
    B = len(blue_points)
    R = len(red_points)
    blue_num = math.ceil(num * B / (B + R))
    red_num = num - blue_num
    blue_index = random.sample(range(B), blue_num)
    red_index = random.sample(range(R), red_num)
    blue_points_sample = []
    red_points_sample = []
    for i in blue_index:
        blue_points_sample.append(blue_points[i])
    for j in red_index:
        red_points_sample.append(red_points[j])

    return blue_points_sample, red_points_sample

def get_nodes(root):
	if root.is_leaf():
		return [root.get_id()]
	else:
		nodes = []
		nodes = nodes + [root.get_id()]
		for child in root.get_children():
			nodes = nodes + get_nodes(child)
		return nodes

def get_cluster_sizes(root):
	if root.is_leaf():
		return []
	sizes = []
	for child in root.get_children():
		sizes = sizes + [child.get_count()]
		sizes = sizes + get_cluster_sizes(child)
	return sizes

def get_leaves(root):
	if root.is_leaf():
		return [root.get_id()]
	else:
		leaves = []
		for child in root.get_children():
			leaves = leaves + get_leaves(child)
		return leaves
def get_children(root):
	children = []
	for child in root.get_children():
		children = children + [child]
	return children

#calculate the upper bound of mw objective using similarity
def get_mw_upper_bound(simi):
    edges = 0.0
    n = simi.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            edges += simi[i][j]
    return edges * (n-2)

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

def print_tree(node, s=""):
    print(s, node.get_id(), node.get_count())
    if not node.is_leaf():
    	for child in node.get_children():
        	print_tree(child, "\t" + s[:-3] + "|--")

def find_maximal_clusters(root,size):
	if root.count > size:
		ids = []
		for child in root.get_children():
			ids = ids + find_maximal_clusters(child, size)
		return ids
	else:
		return [get_leaves(root)]

def get_all_clusters(root):
	if root.is_leaf():
		return []

	clusters = [[]]
	for child in root.get_children():
		clusters = clusters + [get_leaves(child)] + get_all_clusters(child)
	return clusters

def calculate_balance_clusters(clusters, B):
	balance = 1.0
	for cluster in clusters:
		blue = 0
		red = 0
		for u in cluster:
			if u < B:
				blue += 1
			else:
				red += 1
		if blue == 0 or red == 0:
			return 0
		this_balance = np.minimum(float(blue / red),float(red / blue))
		if this_balance < balance:
			balance = this_balance
	return balances

#calculate moseley wang objective function using recursion, returns obj, children
def calculate_hc_obj(simi, root):
	n = simi.shape[0]
	current_nodes = [root]
	obj = 0.0

	while len(current_nodes) > 0:
		parent = current_nodes.pop()
		children = parent.get_children()
		term = n
		if children is None:
			continue
		for child in children:
			current_nodes.append(child)
			term = term - len(get_leaves(child))

		for i in children:
			for j in children:
				if i.get_id() == j.get_id(): continue
				child_i_leaves = get_leaves(i)
				child_j_leaves = get_leaves(j)
				for child_i in child_i_leaves:
					for child_j in child_j_leaves:
						obj += simi[child_i][child_j] * term
	return obj

def calculate_cost_obj(simi,root):
	# n = simi.shape[0]
	obj = 0.0
	leaves = get_leaves(root)
	for i in leaves:
		for j in leaves:
			if i == j: continue
			least_common = lca(root,i,j)
			obj += simi[i][j] * len(get_leaves(least_common))
	return obj

def calculate_revenue_obj(simi,root):
	n = simi.shape[0]
	obj = 0.0
	leaves = get_leaves(root)
	for i in leaves:
		for j in leaves:
			if i == j: continue
			least_common = lca(root,i,j)
			obj += simi[i][j] * (n-len(get_leaves(least_common)))
	return obj

#calculate average similarity between nodes
def average_simi(u,v,j,list_nodes, simi):
    count1 = list_nodes[u].get_count()
    count2 = list_nodes[v].get_count()
    return float((simi[u][j] * count1 + simi[v][j] * count2) / (count1 + count2))

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


#returns an array like pdist
def condense_dist(dist):
    n = dist.shape[0]
    if n == 0 or n == 1:
        return None
    condensed = dist[0][1:]
    for i in range(1,n - 1):
        condensed = np.hstack((condensed, dist[i][i+1:]))
    condensed = np.array(condensed)
    return condensed

def update_simi(simi, left_index, right_index, left_weight, right_weight):
    new_row = (simi[left_index,:] * left_weight + simi[right_index,:] * right_weight) / (left_weight + right_weight)
    simi = np.vstack((simi, new_row))
    new_row = np.append(new_row, 0)
    new_column = new_row.reshape((-1,1))
    simi = np.hstack((simi, new_column))
    simi = np.delete(simi, [left_index, right_index], axis=0)
    simi = np.delete(simi, [left_index, right_index], axis=1)
    return simi

def average_linkage(simi, current_id=None, indices=None, leaves=None):
    n = simi.shape[0]
    if current_id is None:
        current_id = n
    if leaves is None:
        if indices is not None:
            leaves = [Node(id=id, children=None, count=1) for id in indices]
        else:
            leaves = [Node(id=id, children=None, count=1) for id in range(n)]

    while len(leaves) > 1:
        left_index, right_index = find_max(simi, range(len(leaves)))

        left_node = leaves[left_index]
        right_node = leaves[right_index]
        left_weight = left_node.get_count()
        right_weight = right_node.get_count()
        new_node = Node(id=current_id, children=[left_node, right_node],count=left_weight + right_weight)
        current_id += 1
        leaves.append(new_node)
        simi = update_simi(simi, left_index, right_index, left_weight, right_weight)
        del leaves[right_index]
        del leaves[left_index]

    return leaves[0], current_id

def update_simi_max_linkage(simi, left_index, right_index):
    new_row = np.max([simi[left_index,:], simi[right_index,:]],axis=0)
    simi = np.vstack((simi, new_row))
    new_row = np.append(new_row, 0)
    new_column = new_row.reshape((-1,1))
    simi = np.hstack((simi, new_column))
    simi = np.delete(simi, [left_index, right_index], axis=0)
    simi = np.delete(simi, [left_index, right_index], axis=1)
    return simi

def maximum_linkage(simi, current_id=None, indices=None, leaves=None):
    n = simi.shape[0]
    if current_id is None:
        current_id = n
    if leaves is None:
        if indices is not None:
            leaves = [Node(id=id, children=None, count=1) for id in indices]
        else:
            leaves = [Node(id=id, children=None, count=1) for id in range(n)]

    while len(leaves) > 1:
        left_index, right_index = find_max(simi, range(len(leaves)))

        left_node = leaves[left_index]
        right_node = leaves[right_index]

        left_weight = left_node.get_count()
        right_weight = right_node.get_count()
        new_node = Node(id=current_id, children=[left_node, right_node],count=left_weight + right_weight)
        current_id += 1
        leaves.append(new_node)
        simi = update_simi_max_linkage(simi, left_index, right_index)
        del leaves[right_index]
        del leaves[left_index]

    return leaves[0], current_id

##### MAIN ALGORITHMS #####
def rebalance_tree(root):
	if root.is_leaf():
		return root
	v = root
	n = len(get_leaves(root))
	A = get_leaves(max_child(v))
	while len(A) >= (2/3) * n:
		v = max_child(v)
		A = get_leaves(max_child(v))

	root = rebalance_op(root, v, idx=get_max_id(v) + get_max_id(root))
	for child in root.get_children():
		root = remove_child(root, child)
		child = rebalance_tree(child)
		root = add_child(root,child)
	return root

def refine_rebalance(root,eps):
	if root is None:
		return root
	n = len(get_leaves(root))
	if eps <= 1/(2*n) or n == 1:
		return root 

	v = root
	children = get_children(v)
	if len(children) == 1:
		Tbig = max_child(v)
		Tsmall = None
	else:
		Tbig  = max_child(v)
		Tsmall = min_child(v)

	while Tbig.get_count() >= (1/2 + eps) * n:
		temp = Tbig.get_count()
		delta = (Tbig.get_count() - n/2) / n
		Tbig = subtree_search(Tbig, delta * n)
		if temp == Tbig.get_count(): break
	Tbig   = refine_rebalance(Tbig, eps)
	Tsmall = refine_rebalance(Tsmall,eps)

	root.children = [Tbig, Tsmall]
	return root

def fair_hc(root, eps, h, k, B):
	if get_height(root) <= h:
		return level_abstract(root, 0, h)
	depth = math.ceil(math.log2(h))
	V = get_nodes_at(root,depth)
	V = order_by_color(V, B)

	new_root = level_abstract(root, 0, h)

	# if all(child.is_leaf() for child in new_root.get_children()):
	# 	return new_root

	for i in range(k):
		to_fold = []
		for j in range(math.floor(h/k)):
			idx = i + j * k
			to_fold = to_fold + [V[idx]]
		new_root = tree_fold(new_root, to_fold)

	for child in get_children(new_root):
		root = remove_child(root, child)
		child = fair_hc(child, eps, h, k, B)
		root = add_child(root, child)

	return root



##### TREE OPERATORS #####
def rebalance_op(root,v,idx=None):
	if root.get_id() == v.get_id() or v in root.get_children():
		return root
	root = delete_subtree(root,v)
	children = get_children(root)
	counts = 0
	for child in children:
		counts = counts + child.get_count()
		root = remove_child(root,child)
	new_node = Node(id=idx, children=children, count=counts)
	root = add_child(root, v)
	root = add_child(root, new_node)
	return update_counts(root)

def del_ins(root,v,u,idx=None):
	counts = v.get_count() + u.get_count()
	new_node = Node(id=idx, children=[u,v], count=counts)
	root = delete_subtree(root,v)
	root = insert_op(root, new_node, u)
	return update_counts(root)

def level_abstract(root, h1, h2):
	level = 0
	h1_nodes    = get_nodes_at(root, h1)
	k = 0
	while (h2 - k) > h1:
		to_delete = get_nodes_at(root, h2 - k)
		for node in to_delete:
			if node.is_leaf(): continue
			root = delete_node(root, node)
		k = k + 1
	return root 

def tree_fold(root, subtrees):
	if not check_iso(subtrees):
		return root
	for tree in subtrees[1:]:
		root = delete_subtree(root,tree)
		root = fold(root,tree)
	return update_counts(root)

##### HELPER OPERATIONS #####
def order_by_color(nodes, B):
	ratios = []
	for node in nodes:
		rs = 0
		leaves = get_leaves(node)
		for leaf in leaves:
			if leaf >= B: rs = rs + 1
		ratios = ratios + [rs / len(leaves)]
	sort_idx = sorted(range(len(nodes)),key=ratios.__getitem__)
	return [nodes[i] for i in sort_idx]


def get_height(root):
	if root.is_leaf():
		return 0
	else:
		child_heights = []
		for child in root.get_children():
			child_heights = child_heights + [get_height(child)]
		return 1 + max(child_heights)

'''
def fold(root, tree):
	r_children = order_children(root) # get_children(root))
	t_children = order_children(tree) # get_children(tree)
	print_tree(r_children)
	print_tree(t_children)
	print(" ========================= ")
	if len(t_children) == 1 or all(child.is_leaf() for child in t_children):
		root.children = r_children + t_children
		return root

	c_append = []
	for idx in range(len(r_children)):
		if r_children[idx].is_leaf():
			c_append = c_append + [t_children[idx]]
		else:
			r_children[idx] = fold(r_children[idx],t_children[idx])
	r_children = r_children + c_append
	root.children = r_children
	return root
'''

def fold(root, tree):
	if tree.is_leaf():
		root.children = root.get_children() + [tree]
		return root

	r_children = root.get_children()
	t_children = tree.get_children()

	c_append = []
	for idx in range(len(r_children)):
		if not r_children[idx].is_leaf():
			r_children[idx] = fold(r_children[idx],t_children[idx])
		else:
			r_children = r_children + t_children[idx:]
			root.children = r_children
			return root

def delete_node(root, v):
	if root.get_id() == v.get_id():
		return root
	if v in root.get_children():
		root = remove_child(root,v)
		if not (v.get_children() is None):
			root.children = root.children + v.get_children()
	else:
		for child in root.get_children():
			if v.get_id() in get_nodes(child):
				# root  = remove_child(root,child)
				child = delete_node(child,v)
				# root  = add_child(root,child)
	return root

def is_descendant(u, v):
	# Check if v is a descendant of u
	if u.get_id() == v.get_id(): return True
	u_leaves = get_leaves(u)
	v_leaves = get_leaves(v)
	for leaf in v_leaves:
		if leaf not in u_leaves: return False
	return True

def get_nodes_at(root, h):
	if h == 0:
		return [root]

	collected_nodes = []
	if root.get_children() is None:
		return []
	for child in root.get_children():
		collected_nodes = collected_nodes + get_nodes_at(child, h-1)
	return collected_nodes

def lca(root,u,v):
	for child in root.get_children():
		u_descen = u in get_leaves(child)
		v_descen = v in get_leaves(child)
		if u_descen and v_descen:
			return lca(child,u,v)
	return root


def check_iso(subtrees):
	are_leaves = []
	for tree in subtrees:
		are_leaves = are_leaves + [tree.is_leaf()]
	if all(are_leaves): return True
	if any(are_leaves): return False

	for tree in subtrees:
		tree = order_children(tree)

	n = len(subtrees[0].get_children())
	for tree in subtrees:
		if len(tree.get_children()) != n:
			return False

	base_tree     = subtrees[0]
	base_children = get_children(base_tree)
	num_children  = len(base_tree.get_children())
	check = []
	for idx in range(num_children):
		children_set = []
		for tree in subtrees:
			tree_children = get_children(tree)
			children_set = children_set + [tree_children[idx]]
		if not check_iso(children_set):
			return False

	return True

def subtree_search(root, eta):
	v = root
	while min_child(v).get_count() > eta:
		v = max_child(v)
	v = min_child(v)

	u = root
	while max_child(u).get_count() >= v.get_count() and not u.is_leaf():
		u = min_child(u)

	root = del_ins(root, v, u, idx=get_max_id(u) + get_max_id(v))
	return root

def check_balance(root, eps):
	if root.get_count() >= (1/(2*eps)):
		for child in root.get_children():
			c = child.get_count()
			if (1/2 - eps) * root.get_count() > c or c > (1/2 + eps) * root.get_count():
				return False

	if not root.get_children() is None:
		for child in root.get_children():
			return check_balance(child,eps)

	return True

def get_max_id(root): ##
	if root.is_leaf():
		return root.get_id()
	else:
		ids = []
		ids.append(root.get_id())
		for child in root.get_children():
			ids.append(get_max_id(child))
		return max(ids)

def insert_op(root, v, u):
	if u in root.get_children():
		root = remove_child(root,u)
		root = add_child(root,v)
		return root
	else:
		for child in root.get_children():
			if u.get_id() in get_nodes(child):
				root  = remove_child(root, child)
				child = insert_op(child, v, u)
				root  = add_child(root,child)
	return root

def add_child(root, v):
	root.children = get_children(root) + [v]
	return root

def remove_child(root,v):
	root.children.remove(v)
	return root

def max_child(root):
	if root.is_leaf():
		return root
	m_child = None
	max_val = -math.inf
	for child in root.get_children():
		if child.get_count() > max_val:
			max_val = child.get_count()
			m_child = child
	return m_child

def min_child(root):
	if root.is_leaf():
		return root
	min_child = None
	m_child = max_child(root)
	for child in root.get_children():
			if child.get_id() != m_child.get_id():
				min_child = child
	return min_child

def order_children(root):
	children = get_children(root)
	counts = []
	for child in children:
		counts = counts + [child.get_count()]
	sort_idx = sorted(range(len(counts)),key=counts.__getitem__)
	root.children = [children[i] for i in sort_idx]
	return root

def get_id_node(root,id):
	if root.get_id() == id:
		return root

	for child in root.get_children():
		if child.get_id() == id: return child

	for child in root.get_children():
		if id in get_nodes(child):
			return get_id_node(child,id)


def delete_subtree(root, v):
	if v in root.get_children():
		root = remove_child(root,v)
	else:
		for child in root.get_children():
			if v.get_id() in get_nodes(child):
				root  = remove_child(root,child)
				child = delete_subtree(child,v)
				root  = add_child(root,child)

	# Check if singular child to absorb
	children = get_children(root)
	if len(children) == 1: # and not children[0].is_leaf():
		root = children[0]

	return update_counts(root)

def update_counts(root):
	root.count = len(get_leaves(root))
	return root


# ================================================== #

if __name__ == "__main__":
	data = [[1],[2],[3],[7],[8],[9],[4],[5]]#,[11],[13],[18]]
	dist, _ = calculate_distance(data)
	simi = convert_dist(dist)
	root, _ = average_linkage(simi)

