import numpy as np
import heapq
import math
from scipy.spatial import KDTree
from networkx.generators.classic import empty_graph

########################## Robot Utils #######################

def is_standing(mem_queue):
    '''
    mem_queue: memory of robot, where 4,5,6,7th items contain the quaternions of robot's rotation
    '''
    mem = list(mem_queue)

    if len(mem) < 20:
        return True

    for l in mem:
        qx, qy, qz, qw = l[3], l[4], l[5], l[6]
        gz = qx*qx - qy*qy - qz*qz + qw*qw
        if gz > 0:
            return True
    print('Not standing; resetting')

    return False

def is_moving(mem_queue):
    '''
    mem_queue: memory of robot, where last 3 items contain the x,y,z base linear velocities of the robot
    '''
    mem = list(mem_queue)

    if len(mem) < 20:
        return True

    for l in mem:
        speed = np.linalg.norm(l[-3:-1]) # la
        if speed > .4:
            return True

    print('Not moving; resetting')
    return False 

def reached_goal(mem_queue, goal, pos_margin):
    mem = list(mem_queue)

    if len(mem) < 20:
        return False

    for l in mem:
        xy_loc = [l[0], l[1]]
        if np.linalg.norm(np.array(goal) - xy_loc) < pos_margin: 
            print('Reached goal; resetting')
            return True
        
    return False 

def node_path_to_edge_path(path):
    return [(path[n],path[n+1]) for n in range(len(path)-1)]

def forward_cmd(scale):
    return np.array([scale, 0, 0])

def rotate_cmd(direction, scale):
    # direction is 1 for counterclockwise, -1 for clockwise 
    return np.array([scale, 0, direction * scale])

def empty_cmd():
    return np.array([0, 0, 0])

def pure_pursuit_step(path, currentPos, currentHeading, pos_margin = 2, fwd_vel = .5, lookAheadDis=10, LFindex=0, Kp = .5):
  '''
  currentPos: (x,y) loc of robot
  currentHeading: direction of robot, **IN DEGREES**
  pos_margin: radius that robot must achieve around goal node
  fwd_vel: forward velocity of robot
  lookAheadDis: how far robot can find nodes ahead of it (larger = more smooth, but more deviation from path)
  LFIndex: last found node (to make sure always going to the next one)
  Kp: scalar for rotation angle (hyperparameter)
  '''      
    
  # extract currentX and currentY
  currentX = currentPos[0]
  currentY = currentPos[1]

  if np.pi * ((path[-1][0] - currentX)**2 + (path[-1][0] - currentY) ** 2) < pos_margin:
    return 0, 0, 0 

  # use for loop to search intersections
  lastFoundIndex = LFindex
  intersectFound = False
  startingIndex = lastFoundIndex

  for i in range (startingIndex, len(path)-1):

    # beginning of line-circle intersection code
    x1 = path[i][0] - currentX
    y1 = path[i][1] - currentY
    x2 = path[i+1][0] - currentX
    y2 = path[i+1][1] - currentY
    dx = x2 - x1
    dy = y2 - y1
    dr = math.sqrt (dx**2 + dy**2)
    D = x1*y2 - x2*y1
    discriminant = (lookAheadDis**2) * (dr**2) - D**2

    if discriminant >= 0:
      sol_x1 = (D * dy + sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
      sol_x2 = (D * dy - sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
      sol_y1 = (- D * dx + abs(dy) * np.sqrt(discriminant)) / dr**2
      sol_y2 = (- D * dx - abs(dy) * np.sqrt(discriminant)) / dr**2

      sol_pt1 = [sol_x1 + currentX, sol_y1 + currentY]
      sol_pt2 = [sol_x2 + currentX, sol_y2 + currentY]
      # end of line-circle intersection code

      minX = min(path[i][0], path[i+1][0])
      minY = min(path[i][1], path[i+1][1])
      maxX = max(path[i][0], path[i+1][0])
      maxY = max(path[i][1], path[i+1][1])

      # if one or both of the solutions are in range
      if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) or ((minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)):

        foundIntersection = True

        # if both solutions are in range, check which one is better
        if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) and ((minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)):
          # make the decision by compare the distance between the intersections and the next point in path
          if pt_to_pt_distance(sol_pt1, path[i+1]) < pt_to_pt_distance(sol_pt2, path[i+1]):
            goalPt = sol_pt1
          else:
            goalPt = sol_pt2
        
        # if not both solutions are in range, take the one that's in range
        else:
          # if solution pt1 is in range, set that as goal point
          if (minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY):
            goalPt = sol_pt1
          else:
            goalPt = sol_pt2
          
        # only exit loop if the solution pt found is closer to the next pt in path than the current pos
        if pt_to_pt_distance (goalPt, path[i+1]) < pt_to_pt_distance ([currentX, currentY], path[i+1]):
          # update lastFoundIndex and exit
          lastFoundIndex = i
          break
        else:
          # in case for some reason the robot cannot find intersection in the next path segment, but we also don't want it to go backward
          lastFoundIndex = i+1
        
      # if no solutions are in range
      else:
        foundIntersection = False
        # no new intersection found, potentially deviated from the path
        # follow path[lastFoundIndex]
        goalPt = [path[lastFoundIndex][0], path[lastFoundIndex][1]]

    # if determinant < 0
    else:
      foundIntersection = False
      # no new intersection found, potentially deviated from the path
      # follow path[lastFoundIndex]
      goalPt = [path[lastFoundIndex][0], path[lastFoundIndex][1]]

  # obtained goal point, now compute turn vel

  # calculate absTargetAngle with the atan2 function
  absTargetAngle = math.atan2 (goalPt[1]-currentPos[1], goalPt[0]-currentPos[0]) *180/np.pi
  if absTargetAngle < 0: absTargetAngle += 360

  # compute turn error by finding the minimum angle
  turnError = absTargetAngle - currentHeading
  if turnError > 180 or turnError < -180 :
    turnError = -1 * sgn(turnError) * (360 - abs(turnError))
  
  turnError = turnError * np.pi / 180
  # apply proportional controller
  turnVel = Kp*turnError
  
  return fwd_vel, turnVel, lastFoundIndex

def point_traversibilities(points, pos, ahead, up, robot_height=0.3, ahead_tolerance=1, width_tolerance=0.5, height_tolerance=0.3) -> bool :
    '''Determines whether the robot may move forward (whether there is an obstacle within ahead_tolerance of its path)

    Parameters
    points : (n,3), point cloud
    position : 3, robot's position vector (world space) NOTE: assumes points are defined relative to the top center of the robot's back.
    ahead : 3, the robot's ahead direction, unit vector
    up : 3, the up direction TODO: definition of 'up' is still not very clear
    robot_height : float, the height which *measurements* are taken by the robot TODO: 'up'
    ahead_tolerance : the distance which the robot needs to have clear ahead of itself
    width_tolerance : the minimum width by which the robot may traverse
    height_tolerance : the amount of clearance/headspace needed

    Returns
    list : bool, True if point is in the way
    '''
    
    lateral = np.cross(ahead,up)

    assert(np.linalg.norm(lateral)>=.9999 and np.linalg.norm(lateral)<=1.00001) # :)

    # points in robot coordinate system
    points_relative = (points - pos)

    n, _ = points.shape

    ahead_component =   np.linalg.norm((np.tile(points_relative @ ahead, (3,1)).T * np.tile(ahead,(n,1))),axis=1)
    lateral_component = np.linalg.norm((np.tile(points_relative @ lateral, (3,1)).T * np.tile(lateral,(n,1))),axis=1)
    up_component =      np.linalg.norm((np.tile(points_relative @ up, (3,1)).T * np.tile(up,(n,1))),axis=1)
    
    # if all tests are true, point obstructs movement
    test_ahead =    np.abs(ahead_component - ahead_tolerance/2) <= ahead_tolerance # within height
    test_lateral =  np.abs(lateral_component) <= width_tolerance/2 # within width
    test_up =       up_component + robot_height/2 <= height_tolerance # within headspace
    above_ground =  up_component + robot_height >= 0 # above ground

    troublemakers = test_ahead & test_lateral & test_up & above_ground
    
    return troublemakers


############################## Data Structures ################################

class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self) :
        return not self.elements
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]

def stack_if_exists(arr1, arr2, orientation):
    try:
        if orientation == 'h':
            return np.hstack((arr1, arr2))
        elif orientation == 'v':
            return np.vstack((arr1, arr2))
        else:
            raise(Exception('invalid orientation'))
    except:
        # arr1 does not exist
        return arr2
    
def concat_if_exists(arr1, arr2, axis=None):
    try:
        if axis == None:
            return np.concatenate((arr1, arr2))
        else:
            return np.concatenate((arr1, arr2), axis = axis)
    except:
        # arr1 does not exist
        return arr2

################# Edge Costs ####################


def naive_edge_cost(v1, v2, pcl, robot_height):
    '''
    Evaluate the traversability of the edge (v1, v2) based on the density of points in front of the robot along that edge
    '''

    ahead = np.array([v2[0] - v1[0], v2[1] - v1[0], 0])
    if ahead[0] == 0 and ahead[1] == 0:
        return 0
    ahead = ahead / np.linalg.norm(ahead)

    up = np.array([0.0, 0.0, 1.0])
    pos = np.array([v1[0], v1[1], robot_height]) # should z coord be the robot height or 0?
    troublemakers = point_traversibilities(pcl, pos, ahead, up, robot_height=robot_height)
    return sum(troublemakers) # cost is just the number of points that are a problem. All edges are the same distance, so that isn't a problem
    

################################ Math Utils ###################################

def generator_randint(low, high, rng):
    '''
    return a random integer in [low, high)
    '''
    range = high - low
    return math.floor(rng.random() * range + low)

def l2_heuristic(a, b, w=0):
    (x1, y1) = a
    (x2, y2) = b
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def euler_of_quat(quats):
    x = quats[0]
    y = quats[1]
    z = quats[2]
    w = quats[3]
    
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1) * 180 / math.pi
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2) * 180 / math.pi
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4) * 180 / math.pi
    

    return roll_x, pitch_y, yaw_z # in degrees

def triangular_lattice_graph(env_dim):
    x_min, x_max = -env_dim[0] // 2, env_dim[0] // 2
    y_min, y_max = -env_dim[1] // 2, env_dim[1] // 2

    # Define the spacing between the lattice points
    spacing = 1

    # Calculate the number of points along x and y axes
    num_points_x = int((x_max - x_min) / spacing)
    num_points_y = int((y_max - y_min) / spacing)

    # Create an empty graph
    G = empty_graph(0)

    # Generate lattice points and add them as nodes to the graph
    for i in range(num_points_x):
        for j in range(num_points_y):
            x = x_min + i * spacing
            y = y_min + j * spacing
            G.add_node((x, y))

    # Connect the lattice points to form the triangular lattice
    for i in range(num_points_x - 1):
        for j in range(num_points_y - 1):
            x = x_min + i * spacing
            y = y_min + j * spacing

            G.add_edge((x, y), (x + spacing, y + spacing))
            G.add_edge((x, y), (x + spacing, y - spacing))
            G.add_edge((x, y), (x - spacing, y + spacing))
            G.add_edge((x, y), (x - spacing, y - spacing))
            G.add_edge((x,y), (x, y + spacing))
            G.add_edge((x,y), (x+spacing, y))

    return G

def rot_matrix_of_euler(xtheta, ytheta, ztheta):

    c1 = np.cos(xtheta * np.pi / 180)
    s1 = np.sin(xtheta * np.pi / 180)
    c2 = np.cos(ytheta * np.pi / 180)
    s2 = np.sin(ytheta * np.pi / 180)
    c3 = np.cos(ztheta * np.pi / 180)
    s3 = np.sin(ztheta * np.pi / 180)

    matrix=np.array([[c2*c3, -c2*s3, s2],
                [c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1],
                [s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2]])
    
    return matrix

def quat_of_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.
    
    Input
        :param roll: The roll (rotation around x-axis) angle in radians.
        :param pitch: The pitch (rotation around y-axis) angle in radians.
        :param yaw: The yaw (rotation around z-axis) angle in radians.
    
    Output
        :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    
    return (qx, qy, qz, qw)

def pt_to_pt_distance (pt1,pt2):
    distance = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
    return distance


def sgn(num):
  if num >= 0:
    return 1
  else:
    return -1


############################### Other ################################

def unique_name(filename, file_counts):
        '''
        create a prim reference that is not in the set of existing references
        '''
        filename = '/' + filename + '_'
        if filename in file_counts:
            count = file_counts[filename]
            name = filename + str(count+1)
            file_counts[filename] += 1
        else:
            file_counts[filename] = 0
            name = filename + '0'

        return name, file_counts


