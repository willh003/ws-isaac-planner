import math
import numpy as np
import random

import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.spatial import KDTree

from utils import l2_heuristic, PriorityQueue, point_traversibilities
from utils import pure_pursuit_step



class PlanningAgent:
    def __init__(self, start, goal, env_dim=[100,100], nodes_in_graph = 5000):
        '''
        start, goal: (x,y) coords
        env_dim: [x,y] dimension
        
        '''

        edge_radius = 1.5 * (env_dim[0]*env_dim[1] / nodes_in_graph) # radius in which to connect nodes to each other. In practice, 1.5 works well to make sure it is connected but not too dense
        self.graph = EnvironmentGraph(start, goal, env_dim, n_points = nodes_in_graph, edge_radius = edge_radius)
        self.goal = goal
        self.env_dim = env_dim
        self.robot_height = .3
        
    def calculate_path(self, pose, edge_eval):
        '''
        Given a point cloud and the robot's pose, calculate a path to the target
        pcl: nx3 numpy array (given n points)
        pose: tuple of (x, y, z, theta)
        edge_eval: function from two vertices to a cost
        '''
        x, y, z, theta = pose


        # replace w_basic with any cost function on two nodes
        try:
            path = self.graph.shortest_path((x,y), self.goal, l2_heuristic, edge_eval)
            #path = self.graph.shortest_path((x,y), self.goal, l2_heuristic, l2_heuristic)
            # lsp not working for some reason, but whatever
            #path = self.graph.lsp((x,y), self.goal, w_basic, prior_edge_cost, select_forward, l2_heuristic)
            # self.next_node = path.pop()
            self.path = path
        except:
            print('No valid path from current loc: defaulting to previous path')
        return self.path

    def calculate_action(self, path, pose, last_index, fwd_vel, look_ahead, pos_margin = 2, algo='pp'):
        '''
        path: list of tuples corresponding to nodes
        pose: tuple of (x, y, z, theta)
        returns: 3x1 np array, corresopnding to [forward, lateral, rotation] to perform on this timestep
        uses
        '''
        x, y, _, theta = pose

        if algo =='pp':
            return pure_pursuit_step(path, (x, y), 180 - theta * 180 / np.pi, pos_margin=pos_margin, fwd_vel=fwd_vel, lookAheadDis = look_ahead, LFindex = last_index, Kp=.6*fwd_vel)
        else:
            cmd = self.naive_cmd_gen(path[1] if len(path) > 1 else path[0], pose, cmd_scale = fwd_vel)
            return cmd[0], cmd[2], 0


    def node_of_pose(self, G, pose):
        '''
        returns the lower left corner of the robot's current grid cell
        '''
        (x, y, z, theta) = pose
        return (math.floor(x) if x > 0 else 0, math.floor(y) if y > 0 else 0)


    def naive_cmd_gen(self, target_node, pose, ref=np.array([1,0]), rot_margin = .2, pos_margin=.5, cmd_scale = .5):
        '''given a target node, navigate to the node and face that direction
        - rotate until facing the target
        - walk straight in the direction of the target
        ref is the vector corresponding to 0 degree rotation
        rot_margin is the margin of error for the direction of motion (radians)
        pos_margin is the margin of error for the position
        '''
        (x, y, z, theta) = pose
        (xt, yt) = target_node
        pose_to_target = [xt - x, yt-y]
        pose_vector = [np.cos(theta), -np.sin(theta)]

        d_dir = np.sign(pose_to_target[0]*pose_vector[1] - pose_to_target[1]*pose_vector[0]) # pos for cc, neg for c
        d_theta = np.arccos(np.clip(np.dot(pose_to_target , pose_vector) / 
                                        ( np.linalg.norm(pose_to_target) * np.linalg.norm(pose_vector)), -1.0, 1.0))

        # print(pose_to_target / np.linalg.norm(pose_to_target))
        # print(pose_vector / np.linalg.norm(pose_vector))
        # print(d_theta)

        if np.pi * ((xt - x)**2 + (yt - y) ** 2) < pos_margin:
            # if already at n2, pop n1 from path
            self.path = self.path[1:] if len(self.path) > 1 else self.path
            return self.empty_cmd()
        elif d_theta < rot_margin:

            # if pointing towards n2 but not at n2, go forwards
            return self.forward_cmd(cmd_scale)
        else:
            # if not pointing towards n2 and not at n2, rotate to face towards n2
            return self.rotate_cmd(d_dir, cmd_scale)

    def forward_cmd(self, scale):
        return np.array([scale, 0, 0])

    def rotate_cmd(self, direction, scale):
        # direction is 1 for counterclockwise, -1 for clockwise 
        return np.array([scale, 0, direction * scale])

    def empty_cmd(self):
        return np.array([0, 0, 0])


class EnvironmentGraph:
    def __init__(self, start, goal, env_dim, n_points, edge_radius):
        self.adj_list = {}
        self.tree = None
        self.start = start
        self.goal = goal
        self.edge_radius = edge_radius
        
        self.G = self.construct_halton_graph(env_dim, n_points)


    def knn(self, loc, k):
        return self.tree.query(loc, k)

    def add_node(self, loc, neighbors):
        '''
        loc: (x, y) tuple, coordinates in environment of the node
        neighbors: list[(x,y) tuple], coordinates of neighboring nodes
        '''
        for neighbor in neighbors:
            self.G.add_edge(loc, (self.points[neighbor][0], self.points[neighbor][1]))

    def neighbors(self, node):
        return self.G.neighbors(node)

    def construct_halton_graph(self,env_dim, n_points):
        '''
        construct a graph using halton sampling
        ensure start, goal always in graph
        edge_radius is the radius in which to connect nodes to other nodes. Could change this to a gaussian for the future
        '''
        sampler = qmc.Halton(d=2, scramble=True)
        sample = sampler.random(n=n_points)
    
        # put sample into [-env_dim/2, env_dim/2] rectangle    
        sample[:, 0] *= env_dim[0]
        sample[:, 1] *= env_dim[1]
        sample[:, 0] -= env_dim[0] / 2
        sample[:, 1] -= env_dim[1] / 2

        points = np.vstack((sample, np.array([[self.start[0], self.start[1]], [self.goal[0], self.goal[1]]])))
        self.tree = KDTree(points)
        self.points = points

        
        points_in_radius = self.tree.query_ball_point(points, self.edge_radius)

        adj_list = {}
        for i in range(len(points)):
            cur_point = (points[i][0], points[i][1])
            if cur_point not in self.adj_list:
                adj_list[cur_point] = []
            for j in points_in_radius[i]:
                adj_point = (points[j][0], points[j][1])
                
                # no self loops
                if adj_point != cur_point:
                        adj_list[cur_point].append(adj_point)

        G = nx.Graph([(key, item) for key in adj_list.keys() for item in adj_list[key] ])
        return G

    def shortest_path(self, start, goal, heuristic, cost_fn):
        '''
        start, goal: (x, y) tuples representing points in space for nodes
        cost_fn: function on (v1, v2), returning cost of traversing an edge (v1, v2)
        '''

        # make sure the current location is in the graph (we could replan from anywhere in the environment)
        if start not in self.G.nodes():
            self.add_node(start, neighbors = self.tree.query_ball_point(start, self.edge_radius))   
        if goal not in self.G.nodes():
            self.add_node(goal, neighbors = self.tree.query_ball_point(goal, self.edge_radius))   

        path = nx.astar_path(self.G, start, goal, heuristic=l2_heuristic, weight = lambda v1, v2, attr: cost_fn(v1, v2))
        return path

    def lsp(self, start, goal, w, w_est, selector, heuristic):
        '''
        Implementation of LazySP path finding algorithm. Aims to minimize edge evaluations required
        w is a function which returns the cost of each edge in G (high computation cost)
        w_est is a prior on w
        selector is a function which establishes the rule for which edge to evaluate next
        heuristic is used for A* pathfinding
        https://www.ri.cmu.edu/pub_files/2016/3/paper-lazysp.pdf 
        '''
        E_eval = set()
        w_lazy = {}
        
        for (v1, v2) in self.G.edges:
            w_lazy[(v1, v2)] = w_est((v1, v2), heuristic) # replace with w_est[edge], if you have a prior on the edges
            w_lazy[(v2, v1)] = w_lazy[(v1, v2)] # assume an undirected graph (direction of traverse shouldn't matter)

        attempts = 0
        while attempts < 200:
            p_candidate = self.shortest_path(start, goal, heuristic, cost_fn=lambda v1, v2: w_lazy[(v1, v2)])

            if already_evaluated(p_candidate, E_eval):
                return p_candidate
            
            E_selected = selector(self.G, E_eval, p_candidate)
            for e in E_selected:
                
                if e not in E_eval:
                    w_lazy[e] = w(e) # w is a function which evaluates the cost of an edge in the graph
                    E_eval.add(e)

        raise Exception('Did not find a shortest path')

def reconstruct_path(came_from,
                     start, goal):

    current= goal
    path = []
    if goal not in came_from: # no path was found
        return []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start) # optional
    path.reverse() # optional
    return path

def astar_tuple(G, start, goal, heuristic, cost):
    '''
    G: a graph object with a function neighbors(self, node), returning an iterable of neighbors of any node in G
    run A* on a graph where nodes are represented by (x, y), denoting their location in space
    heuristic and cost are functions (e1, e2) -> int, where e1 =(x1, y1)
    heuristic represents distance from goal node, cost represents distance between two nodes
    '''
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from= {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    goal_pos = goal
    
    while not frontier.empty():
        current = frontier.get()
        if current == goal:
            #print('success')
            break
        
        for next in G.neighbors(current):

            new_cost = cost_so_far[current] + cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(next, goal_pos)
                frontier.put(next, priority)
                came_from[next] = current
    
    return came_from, cost_so_far  




############################################# LSP, Spline (extensions) ################################################

def spline_cmd_gen(n1, n2, pose):
    '''
    given start node, end node, and pose
    sample a node n3 in a straight line from the pose (or maybe a few nodes)
    generate a spline using n1, n2 and n3
    travel a small distance along the spline
        - let current pose by v
        - find its gradient g
        - let rot = k*(g-v)
        - otherwise, just use forward command
    recompute
    '''


def already_evaluated(path, E_eval):
    for n in range(len(path)-1):
        e = (path[n],path[n+1]) 
        if e not in E_eval:
            return False
    return True

def select_forward(G, E_eval, path):
    for n in range(len(path)-1):
        e = (path[n],path[n+1]) 
        if e not in E_eval:
            return [e]
    raise Exception('All edges in path have already been evaluated, but a path was not returned')

def w_basic(e):
    return 1

def w_rand(e):
    return np.random.rand()

def w_adversary(e):
    if e == ((2,1), (3,1)):
        return 100
    else:
        return 1

def w_pcl(e, pcl):
    pass

def prior_edge_cost(e, heuristic):
    '''
    get the estimated cost for an edge (without evaluation)
    '''
    return heuristic(e[0], e[1])




def test1():
    G = nx.grid_graph(dim=[20,20], periodic=True)
    # G = nx.hexagonal_lattice_graph(20, 20)
    # G = nx.connected_watts_strogatz_graph(20, 4, .1)
    start = (1,1)
    goal = (18,15)

    # came_from, cost_so_far = astar_tuple(G, start, goal, l2_heuristic)
    # path1 = reconstruct_path(came_from, start, goal)
    # edge_path1 = [(path1[n],path1[n+1]) for n in range(len(path1)-1)]
    #path_lsp = lsp(G, start, goal, w_basic, prior_edge_cost, select_forward, l2_heuristic)
    #edge_path_lsp = [(path_lsp[n],path_lsp[n+1]) for n in range(len(path_lsp)-1)]

    path = nx.astar_path(G, start, goal, heuristic=l2_heuristic, weight=l2_heuristic)
    edge_path =  [(path[n],path[n+1]) for n in range(len(path)-1)]
   # print(edge_path_lsp)

    pos = {(x,y):(y,-x) for x,y in G.nodes}
    #nx.draw(G,pos,node_color='k')
    nx.draw_networkx_nodes(G,pos=pos, node_size=25)

    nx.draw_networkx_edges(G,pos=pos,edgelist=G.edges,edge_color = 'b', width=1)
    nx.draw_networkx_edges(G,pos=pos,edgelist=edge_path,edge_color = 'r', width=3)
   # nx.draw_networkx_edges(G,pos=pos,edgelist=edge_path_lsp,edge_color = 'g', width=5)

    plt.show()

def test2():

    planner = PlanningAgent(goal=(81, 80, 0, 1.2), env_dim=[100, 100])
    pcl = np.loadtxt("depth.csv", delimiter=",")
    pose = (1.5, 1.2, 0, 1.2)

    return planner.get_action(pcl, pose)


def test3():



    planner = PlanningAgent((0,0), (95, 93), [100, 100])

    path = planner.calculate_path([], (0,0,0,0), l2_heuristic)
    print(path)

    #G = EnvironmentGraph((0,0), (95, 93), [100, 100], 800, 6)
    #path = G.shortest_path(G.start, G.goal, l2_heuristic, l2_heuristic)
   # print(path)
    # edge_path =  [(path[n],path[n+1]) for n in range(len(path)-1)]

   # path_lsp = G.lsp(G.start, G.goal, w_basic, prior_edge_cost, select_forward, l2_heuristic)
   # print(path_lsp)
    # edge_path_lsp =  [(path_lsp[n],path_lsp[n+1]) for n in range(len(path_lsp)-1)]

    # pos = {(x,y):(x,y) for x,y in G.G.nodes}
    # #nx.draw(G,pos,node_color='k')
    # nx.draw_networkx_nodes(G.G,pos=pos, node_size=25)
    # nx.draw_networkx_edges(G.G,pos=pos,edgelist=edge_path,edge_color = 'r', width=8)
    # nx.draw_networkx_edges(G.G,pos=pos,edgelist=edge_path_lsp,edge_color = 'g', width=5)
    # nx.draw_networkx_edges(G.G,pos=pos,edgelist=G.G.edges,edge_color = 'b', width=1)
    #plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    test3()

