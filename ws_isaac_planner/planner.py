
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.spatial import KDTree
from ws_isaac_planner.utils import *


MARKER_LEN = 12
class PlanningAgent:
    def __init__(self, start, goal, env_dim=[100,100], map_resolution = 4, edge_radius = 2 ,draw_markers = ''):
        '''
            start: (x,y) coords
            
            goal: (x,y) coords
            
            env_dim: [x,y] dimension of world

            map_resolution: cells in map per unit length in the world 
            
            edge_radius: radius in which to connect neighbors to new nodes in the graph (too small will cause errors)

            draw_markers: 'dense', 'sparse', or '' (which won't draw anything)
        '''

        # 2D lattice graph with a node every unit length in the world
        self.graph = EnvironmentGraph(start, goal, env_dim, edge_radius = edge_radius)
        
        # traversability map, starts at 1 everywhere so it is optimistic (encourages exploration)
        self.global_map = torch.ones(env_dim[0] * map_resolution, env_dim[1] * map_resolution) 
        self.map_resolution = map_resolution

        self.goal = goal
        self.env_dim = env_dim
        self.last_index = 0
        self.path = None
        self.draw_markers = draw_markers
        self._rng = np.random.default_rng()

        if self.draw_markers == 'dense' or self.draw_markers == 'sparse':
            self.draw_graph_markers()

    def update_global_map(self, trav, pixels_in_world, method = 'exp', alpha=.8):
        '''
        Inputs:
            trav: torch.tensor, shape (height, width), representing traversability signal from model

            pixels_in_world: torch.tensor, shape (3,height, width), such that pixels_in_world[:,i,j] is the xyz coords of pixel i,j 

            method: 'exp' or 'recent'. Specifies the method for handling map collisions. exp uses an exponential moving average 

            alpha: specifies the amount to weigh the most recent observation by for exponential averaging
        Behavior:
            sets global map traversabilities according to coordinates and method
        '''
        trav= trav.to('cpu')
        pixels_in_world = pixels_in_world.to('cpu').long()

        
        x_coordinates = pixels_in_world[0, :, :]
        y_coordinates = pixels_in_world[1, :, :]
        z_coordinates = pixels_in_world[2, :, :]

        x_bound = self.env_dim[0]
        y_bound = self.env_dim[1]


        bound_mask = (x_coordinates < -x_bound) | (x_coordinates > x_bound) | (y_coordinates < -y_bound) | (y_coordinates > y_bound)
        nan_mask = torch.isnan(x_coordinates) | torch.isnan(y_coordinates) | torch.isnan(z_coordinates)
        good_pts = ~(bound_mask | nan_mask)
        
        x_coordinates = x_coordinates[good_pts]
        y_coordinates = y_coordinates[good_pts]

        trav_inbounds = trav[good_pts]

        if method == 'exp':
            self.global_map[x_coordinates, y_coordinates] = alpha * trav_inbounds + (1-alpha) * self.global_map[x_coordinates, y_coordinates]
        elif method == 'recent':
            self.global_map[x_coordinates, y_coordinates] = trav


    def random_new_goal(self, margin = 5):
        x = self.env_dim[0] // 2
        y = self.env_dim[1] // 2

        potential_goals = [(generator_randint(-x, x, self._rng), generator_randint(y-margin,y, self._rng)), 
                           (generator_randint(x-margin, x, self._rng), generator_randint(-y,y, self._rng)), 
                           (generator_randint(-x, -x + margin, self._rng), generator_randint(-y,y, self._rng)), 
                            (generator_randint(-x, x, self._rng), generator_randint(-y,-y+margin, self._rng))]
        
        new_goal = potential_goals[generator_randint(0,3, self._rng)]

        self.goal = new_goal
        self.graph.goal = new_goal

        print(f'New goal: {new_goal}')

    def calculate_path(self, pose, edge_method='l2'):
        '''
        Given the robot's pose and a function returning the cost, calculate a path to the target
        arraylike, where first two elements are x,y
        edge_method: method for evaluating edges. 'l2' for euclidean distance, 'trav-map' for traversability map
        '''
        x = pose[0]
        y = pose[1]

        if edge_method == 'l2':
            edge_eval = l2_heuristic
        elif edge_method == 'trav-map':
            def edge_eval(v1, v2):
                return cost_from_grid((v1,v2), self.global_map, 
                               lambda loc: (int(loc[0] * self.map_resolution), int(loc[1] * self.map_resolution)))
        else:
            raise Exception(f'invalid argument for edge_method {edge_method}')

        try:
            path = self.graph.shortest_path((x,y), self.goal, l2_heuristic, edge_eval)
            self.path = path
            self.last_index = 0

            if self.draw_markers == 'dense' or self.draw_markers == 'sparse':
                self.replan_markers()
        except:
            # networkx raises an error if no path can be found
            print('No path found, reverting to old path')

        return self.path

    def calculate_action(self, path, pose, fwd_vel, look_ahead, pos_margin = 2, algo='pp'):
        '''
        Inputs:
            path: list of tuples corresponding to nodes
        
            pose: arraylike, where first three elements are x,y, theta
            
        Returns: 
            3x1 np array, corresopnding to [forward, lateral, rotation] to perform on this timestep
        
        '''
        
        if not path:
            path = self.calculate_path(pose) # naively calculate path at first

        x = pose[0]
        y = pose[1]
        theta = pose[2]

        # if running pure pursuit
        if algo =='pp':
            fwd, rot, last_index = pure_pursuit_step(path, (x, y), 180 - theta * 180 / np.pi, pos_margin=pos_margin, fwd_vel=fwd_vel, lookAheadDis = look_ahead, LFindex = self.last_index, Kp=.6*fwd_vel)
            self.last_index = last_index
            return [fwd, 0, rot]
        else:
            return self.naive_cmd_gen(path[1] if len(path) > 1 else path[0], pose, cmd_scale = fwd_vel)

    def draw_graph_markers(self):
        from omni.isaac.orbit.markers import PointMarker
        nodes = self.graph.G.nodes
        self.node_to_idx = {}
        poses = np.zeros((len(nodes), 3))
        i=0
        for node in nodes:
            poses[i, 0] = node[0]
            poses[i, 1] = node[1]
            poses[i, 2] = 0 
            self.node_to_idx[node] = i
            i+=1
        
        if self.draw_markers == 'sparse':
            self.markers = PointMarker(f"/World/graph", MARKER_LEN, radius=0.3)
        else:
            self.markers = PointMarker(f"/World/graph", len(nodes), radius=0.15)
            self.markers.set_world_poses(poses)

    def replan_markers(self):
        path = self.path
        indices = []

        marker_poses = np.zeros((MARKER_LEN, 3))

        for i in range(len(path)):
            node = path[i]
            if i < MARKER_LEN:
                marker_poses[i][0] = node[0]
                marker_poses[i][1] = node[1]

            if node in self.node_to_idx:
                indices.append(self.node_to_idx[node])

        if self.draw_markers == 'dense':
            self.markers.set_status(np.array([0]*self.markers.count))        
            self.markers.set_status(np.array([1]*len(indices)), indices = indices)
        else:        
            self.markers.set_world_poses(marker_poses)
            self.markers.set_status(np.array([1]*MARKER_LEN))  

    def naive_cmd_gen(self, target_node, pose, ref=np.array([1,0]), rot_margin = .2, pos_margin=.5, cmd_scale = .5):
        '''
        @param ref: the vector corresponding to 0 degree rotation
        @param rot_margin: the margin of error for the direction of motion (radians)
        @param pos_margin: the margin of error for the position
        
        given a target node:
            navigate to the node and face that direction
            rotate until facing the target
            walk straight in the direction of the target
        '''
    
        (x, y, z, theta) = pose
        (xt, yt) = target_node
        pose_to_target = [xt - x, yt-y]
        pose_vector = [np.cos(theta), -np.sin(theta)]

        d_dir = np.sign(pose_to_target[0]*pose_vector[1] - pose_to_target[1]*pose_vector[0]) # pos for cc, neg for c
        d_theta = np.arccos(np.clip(np.dot(pose_to_target , pose_vector) / 
                                        ( np.linalg.norm(pose_to_target) * np.linalg.norm(pose_vector)), -1.0, 1.0))

        if np.pi * ((xt - x)**2 + (yt - y) ** 2) < pos_margin:
            # if already at n2, pop n1 from path
            self.path = self.path[1:] if len(self.path) > 1 else self.path
            return empty_cmd()
        elif d_theta < rot_margin:

            # if pointing towards n2 but not at n2, go forwards
            return forward_cmd(cmd_scale)
        else:
            # if not pointing towards n2 and not at n2, rotate to face towards n2
            return rotate_cmd(d_dir, cmd_scale)



class EnvironmentGraph:
    def __init__(self, start, goal, env_dim, edge_radius):
        self.adj_list = {}
        self.tree = None
        self.start = start
        self.goal = goal
        self.edge_radius = edge_radius # radius in which to add neighbors to a new node
        
        self.G = self.construct_lattice_graph(env_dim)

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

    def construct_lattice_graph(self, env_dim):

        graph = triangular_lattice_graph(env_dim)
        
        points = np.zeros((len(graph.nodes),2))

        i=0
        for x, y in graph.nodes:
            points[i][0] = x
            points[i][1] = y
            i+=1
        
        self.tree = KDTree(points)
        self.points = points
        return graph

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
            start_neighbors = self.tree.query_ball_point(start, self.edge_radius)
            if len(start_neighbors) == 0:
                print('no start neighbors found')
            self.add_node(start, neighbors = start_neighbors) 
        if goal not in self.G.nodes():
            if len(start_neighbors) == 0:
                print('no goal neighbors found')
            goal_neighbors = self.tree.query_ball_point(goal, self.edge_radius)
            self.add_node(goal, neighbors = goal_neighbors) 

        # pos = {(x,y):(x,y) for x,y in self.G.nodes}
        # nx.draw_networkx_nodes(self.G,pos=pos, node_size=25)

        # #nx.draw_networkx_edges(G.G,pos=pos,edgelist=edge_path_l,edge_color = 'g', width=5)
        # nx.draw_networkx_edges(self.G,pos=pos,edgelist=self.G.edges,edge_color = 'b', width=1)
        # plt.savefig('graph.png')

        path = nx.astar_path(self.G, start, goal, heuristic=heuristic, weight = lambda v1, v2, attr: cost_fn(v1, v2))
        
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



    planner = PlanningAgent((.4,.3), (7.5,24), env_dim=[50, 50], nodes_in_graph=500)

    path = planner.calculate_path([0,0], edge_eval=l2_heuristic)

    #G = EnvironmentGraph((0,0), (95, 93), [100, 100], 800, 6)
    #path = G.shortest_path(G.start, G.goal, l2_heuristic, l2_heuristic)
   # print(path)
    # edge_path =  [(path[n],path[n+1]) for n in range(len(path)-1)]

   # path_lsp = G.lsp(G.start, G.goal, w_basic, prior_edge_cost, select_forward, l2_heuristic)
   # print(path_lsp)

    #nx.draw(G,pos,node_color='k')
    pos = {(x,y):(x,y) for x,y in planner.graph.G.nodes}
    nx.draw_networkx_nodes(planner.graph.G,pos=pos, node_size=25)

    #nx.draw_networkx_edges(G.G,pos=pos,edgelist=edge_path_l,edge_color = 'g', width=5)
    nx.draw_networkx_edges(planner.graph.G,pos=pos,edgelist=planner.graph.G.edges,edge_color = 'b', width=1)
    try:
        edge_path =  [(path[n],path[n+1]) for n in range(len(path)-1)]


        nx.draw_networkx_edges(planner.graph.G,pos=pos,edgelist=edge_path,edge_color = 'r', width=8)
    except:
        print('fail')

    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    test3()

