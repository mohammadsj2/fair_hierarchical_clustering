#this file is the helper file for the average linkage tracking
import numpy as np
import math
import time
import random

def inter_fairlet_simi(simi, fairlets):
    m = len(fairlets)
    fairlets_simi = np.zeros((m, m))
    fairlets_flatten = []
    for i in range(m):
        x = []
        x.extend(fairlets[i][0])
        x.extend(fairlets[i][1])
        fairlets_flatten.append(x)

    for i in range(1,m):
        for j in range(i):
            fairlets_simi[i][j] = np.sum(np.sum(simi[fairlets_flatten[i]][:,fairlets_flatten[j]]))
            fairlets_simi[j][i] = fairlets_simi[i][j]
    return fairlets_simi

def inter_fairlet_simi_multi_color(simi, fairlets):
    m = len(fairlets)
    color_types = len(fairlets[0])
    fairlets_simi = np.zeros((m, m))
    fairlets_flatten = []
    for i in range(m):
        x = []
        for color in range(color_types):
            x.extend(fairlets[i][color])
        fairlets_flatten.append(x)

    for i in range(1, m):
        for j in range(i):
            fairlets_simi[i][j] = np.sum(np.sum(simi[fairlets_flatten[i]][:,fairlets_flatten[j]]))
            fairlets_simi[j][i] = fairlets_simi[i][j]
    return fairlets_simi

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

class Node:
    def __init__(self, id = None, left = None, right = None, count = 0):
        self.id = id
        self.left = left
        self.right = right
        self.count = count

    def get_count(self):
        return self.count

    def get_id(self):
        return self.id

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    def is_leaf(self):
        if self.left == None and self.right == None:
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


def get_children(root):
    if root.is_leaf():
        return [root.get_id()]
    else:
        return get_children(root.get_left()) + get_children(root.get_right())


def print_tree(node, s=""):
    print(s, node.get_id(), node.get_count())
    if not node.is_leaf():
        print_tree(node.get_left(), "\t" + s[:-3] + "|--")
        print_tree(node.get_right(), "\t" + s[:-3] + "\\--")

def find_maximal_clusters(root,size):
    if root.count > size:
        return find_maximal_clusters(root.get_left(), size) + find_maximal_clusters(root.get_right(), size)
    else:
        return [get_children(root)]

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
    return balance

#calculate moseley wang objective function using recursion, returns obj, children
def calculate_hc_obj(simi, root):
    n = simi.shape[0]
    current_nodes = [root]
    obj = 0.0
    while len(current_nodes) > 0:
        parent = current_nodes.pop()
        left_child = parent.left
        right_child = parent.right
        if left_child is None or right_child is None:
            continue
        current_nodes.append(left_child)
        current_nodes.append(right_child)

        left_children = get_children(left_child)
        right_children = get_children(right_child)
        term = n - len(left_children) - len(right_children)

        for i in left_children:
            for j in right_children:
                obj += simi[i][j] * term

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
            leaves = [Node(id=id, left=None, right=None, count=1) for id in indices]
        else:
            leaves = [Node(id=id, left=None, right=None, count=1) for id in range(n)]

    while len(leaves) > 1:
        left_index, right_index = find_max(simi, range(len(leaves)))

        left_node = leaves[left_index]
        right_node = leaves[right_index]
        left_weight = left_node.get_count()
        right_weight = right_node.get_count()
        new_node = Node(id=current_id, left=left_node, right=right_node,count=left_weight + right_weight)
        current_id += 1
        leaves.append(new_node)
        simi = update_simi(simi, left_index, right_index, left_weight, right_weight)
        del leaves[right_index]
        del leaves[left_index]

    return leaves[0], current_id

#write an average linkage algorithm with fairlets
def avlk_with_fairlets(simi, fairlets):
    fairlet_roots = []
    n = simi.shape[0]
    m = len(fairlets)
    current_id = n
    for y in fairlets:
        x = []
        x.extend(y[0])
        x.extend(y[1])
        this_root, current_id = average_linkage(simi=simi[x][:,x], current_id=current_id, indices=x)
        fairlet_roots.append(this_root)

    fairlet_simi = inter_fairlet_simi(simi, fairlets)
    for i in range(m):
        for j in range(m):
            fairlet_simi[i][j] = fairlet_simi[i][j] / (fairlet_roots[i].get_count() * fairlet_roots[j].get_count())
    root, _ = average_linkage(simi=fairlet_simi, current_id = current_id, leaves = fairlet_roots)
    return root

def avlk_with_fairlets_multi_color(simi, fairlets):
    color_types = len(fairlets[0])
    fairlet_roots = []
    n = simi.shape[0]
    m = len(fairlets)
    current_id = n
    for y in fairlets:
        x = []
        for color in range(color_types):
            x.extend(y[color])
        this_root, current_id = average_linkage(simi=simi[x][:,x], current_id=current_id, indices=x)
        fairlet_roots.append(this_root)

    fairlet_simi = inter_fairlet_simi_multi_color(simi, fairlets)
    for i in range(m):
        for j in range(m):
            fairlet_simi[i][j] = fairlet_simi[i][j] / (fairlet_roots[i].get_count() * fairlet_roots[j].get_count())
    root, _ = average_linkage(simi=fairlet_simi, current_id = current_id, leaves = fairlet_roots)
    return root

### ======================================== ###
###         Springer added functions:        ###
### ======================================== ###

def tree_rebalance(root): # WHY IS THERE AN INFINITE RECURSION HERE??
    '''
    Input:  tree hierarchy with larger clusters always on right
    Output: tree with relative balance of 1/6
    '''
    if root.is_leaf() or root is None:
        return root
    n = count_nodes(root) #root.get_count()
    v = root
    # r = root
    A = (v.get_right()).get_count()
    print(n)
    print(A)
    while A >= (2/3) * n:
        print("reached")
        v = v.get_right()
        A = (v.get_right()).get_count()
    tree = rebalance_operator(root, v)
    tree_left  = tree_rebalance(tree.get_left())
    tree_right = tree_rebalance(tree.get_right())

    tree.left  = tree_left
    tree.right = tree_right
    return tree

def tree_refine_rebalance(root, eps):
    '''
    Input  : 1/6-balanced tree and parameter eps in (0,1/6)
    Output : eps-balanced tree
    '''
    current_nodes = [root]
    n = len(current_nodes)
    if eps >= (1/2) * n:
        return root
    v = current_nodes.pop()
    tree_big   = v.left
    tree_small = v.right
    while len(tree_big) >= (1/2 + eps)*n:
        delta = (len(tree_big) - (n/2))/n
        tree_big = subtree_search(tree_big, delta*n)
    tree_big   = tree_refine_rebalance(tree_big, eps)
    tree_small = tree_refine_rebalance(tree_small, eps)

    tree.left  = tree_big
    tree.right = tree_small
    return tree

def subtree_search(root, err):
    '''
    Input  : tree and error parameter
    Output : 
    '''
    current_nodes = [root]
    v = current_nodes.pop()
    while len(v.left) > err:
        v = v.left
    v = v.right

    u = root
    while len(u.left) >= len(v):
        u = u.right
    tree = del_ins(u,v) # need to fix del ins to be called on an overall tree?
    return tree

def fair_hc(root, h, k):
    '''
    Input  : eps relatively balanced tree with parameters h = 2^i, k = 2^j
    Output : fair tree T'
    '''
    print("unfinished")

    # TODO

def get_tree_height(root, v):
    # TODO
    print("unfinished")


# ------ TREE OPERATORS ------- #
def count_nodes(root):
    if root is None:
        return 0
    elif root.is_leaf():
        return 1
    else:
        return 1 + count_nodes(root.get_left()) + count_nodes(root.get_right())

def get_node_depth(root, v):
    if root is None or root.get_id() == v.get_id():
        return 0
    # elif (root.get_left()).get_id() == v.get_id() or (root.get_right()).get_id() == v.get_id():
    #     return 1
    else:
        return 1 + min(get_node_depth(root.get_left(),v),get_node_depth(root.get_right(),v))

def get_max_id(root): ##
    if root.is_leaf():
        return root.get_id()
    else:  
        right = root.get_right()
        left  = root.get_left()
        return max(root.get_id(), get_max_id(right), get_max_id(left))

def reset_count(root): ##
    if root is None:
        return root
    root.count = len(get_children(root))
    root.left  = reset_count(root.get_left())
    root.right = reset_count(root.get_right())
    return root

def delete_operator(root,v): ##
    '''
    Remove subtree rooted at v and return tree
    '''
    if root.get_id() == v.get_id() or root.is_leaf():
        return root
    elif root.left.get_id() == v.get_id():
        return root.get_right()
    elif root.right.get_id() == v.get_id():
        return root.get_left()
    else:
        root.left = delete_operator(root.left, v)
        root.right = delete_operator(root.right, v)
    # UPDATE COUNTS
    root = reset_count(root)
    return root


def insert_operator(root,v,u): ##
    '''
    Inserts node v "at" node u in the tree root
    '''
    if root.get_id() == u.get_id():
        return Node(id=None, left=u, right=v, count=u.get_count() + v.get_count())
    elif root.get_left().get_id() == u.get_id():
        parent = Node(id=None, left=u, right=v, count=u.get_count() + v.get_count())
        root.left = parent
    elif root.get_right().get_id() == u.get_id():
        parent = Node(id=None, left=u, right=v, count=u.get_count() + v.get_count())
        root.right = parent
    root = reset_count(root)
    return(root)

def check_isomorph(u,v):
    if u.is_leaf() and v.is_leaf():
        return True
    elif u.is_leaf():
        return False
    elif v.is_leaf():
        return False
    else:
        return check_isomorph(u.get_right(),v.get_right()) and check_isomorph(u.get_left(),v.get_left())

def abstract_operator(h1,h2):
    # TODO
    print("unfinished")

def fold_operator(ts):
    # TODO
    print("unfinished")

def rebalance_operator(root, v): ##
    '''
    Reset v's children to be u and (new node) c
    '''
    if root.get_id() == v.get_id():
        return root
    # root_del_v = delete_operator(root,v)
    root = delete_operator(root,v)
    left_tree  = root.get_left()
    right_tree = root.get_right()

    if left_tree is not None and right_tree is not None:
        root.left = Node(id=None, left=left_tree, right=right_tree, count=left_tree.get_count() + right_tree.get_count())
    elif left_tree is not None and right_tree is None:
        root.left = Node(id=None, left=left_tree, right=right_tree, count=left_tree.get_count())
    elif left_tree is None and right_tree is not None:
        root.left = Node(id=None, left=left_tree, right=right_tree, count=right_tree.get_count())
    else:
        root.left = Node(id=None, left=left_tree, right=right_tree, count=0)
    root.right = v

    return reset_count(root)


if __name__ == "__main__":
    data = [[1],[2],[3],[7],[8],[9],[4],[5]]
    dist, _ = calculate_distance(data)
    simi = convert_dist(dist)
    root, _ = average_linkage(simi)

    print_tree(root)
    root = tree_rebalance(root)
    print_tree(root)

    #bal_root = tree_rebalance(root)
    #print_tree(bal_root)
    #print(get_mw_upper_bound(simi))
    #print(calculate_hc_obj(simi, root))
