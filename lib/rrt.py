import numpy as np
import random
from lib.calculateFK import FK
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap
from copy import deepcopy


def getRandomConfiguration(lowerLim, upperLim):
    """
    Generate a random robot configuration within joint limits.
    :param lowerLim: Lower joint limits.
    :param upperLim: Upper joint limits.
    :return: Random robot configuration.
    """
    return np.random.uniform(lowerLim, upperLim)


def isRobotCollided(q_from, q_to, obstacles):
    #
    #obstacle - nx6 numpy array representing the obstacle box min and max positions
    #in the world frame
    #
    fk = FK()
    j_from, _ = fk.forward(q_from) # joints 8x3
    j_to, _ = fk.forward(q_to) # joints 8x3

    n_o = len(obstacles)
    tol = np.array([-0.05, -0.05, -0.05, 0.05, 0.05, 0.05])

    for i in range(n_o):
        if np.any(detectCollision(j_from, j_to, obstacles[i] + tol)):
            return True
        intermediate_points = np.linspace(q_from, q_to, num=10, axis=1)
        for j in range(intermediate_points.shape[1]-1):
            j_1, _ = fk.forward(intermediate_points[..., j])
            j_2, _ = fk.forward(intermediate_points[..., j+1])
            if np.any(detectCollision(j_1, j_2, obstacles[i] + tol)):
                return True
    # check for self collision_tolerance
    distance_matrix = np.sqrt(((j_to[:, np.newaxis] - j_to)**2).sum(axis=2))
    upper_triangle_distances = np.triu(distance_matrix, k=1)
    upper_triangle_flat = upper_triangle_distances[upper_triangle_distances != 0]
    if np.any(upper_triangle_flat < 0.1):
        return True

    return False

def findNearsetNode(q, tree):
    """
    Find the nearest node in the tree to the query node q.

    :param q: Joints variables (1x7 vector).
    :param tree: A list of nodes, where each node is a joints variable (nx7 matrix).
    :return: Index of the nearest node in the tree to q.
    """
    tree_arr = np.array(tree)
    distances = np.linalg.norm(tree_arr - q, axis=1)
    nearest_index = np.argmin(distances)

    return nearest_index, tree[nearest_index]


def rrt(map, start, goal):
    """
    Implement RRT algorithm in this file.
    :param map:         the map struct
    :param start:       start pose of the robot (0x7).
    :param goal:        goal pose of the robot (0x7).
    :return:            returns an mx7 matrix, where each row consists of the configuration of the Panda at a point on
                        the path. The first row is start and the last row is goal. If no path is found, PATH is empty
    """

    # initialize path
    path = []
    T_start = [start]
    P_start = [None] # parent vector for start tree
    T_goal = [goal]
    P_goal = [None] # parent vector for goal tree
    obstacles = map.obstacles

    max_iter = 500
    # get joint limits
    lowerLim = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upperLim = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])
    cnt = 0
    is_path = False
    while cnt < max_iter:
        q_samp = getRandomConfiguration(lowerLim, upperLim)
        i_st, q_st = findNearsetNode(q_samp, T_start)
        connect = True
        if not isRobotCollided(q_st, q_samp, obstacles):
            T_start.append(q_samp)
            P_start.append(i_st)
        else:
            connect = False
        i_g, q_g = findNearsetNode(q_samp, T_goal)
        if not isRobotCollided(q_g, q_samp, obstacles):
            T_goal.append(q_samp)
            P_goal.append(i_g)
        else:
            connect = False
        if connect:
            is_path = True
            break
        cnt+=1

    if is_path:
        q = T_start[-1]
        p_i = P_start[-1]
        path.append(q)
        while p_i is not None:
            q_p = T_start[p_i]
            path.append(q_p)
            p_i = P_start[p_i]
        path.reverse() # reverse the list
        p_i = P_goal[-1]
        while p_i is not None:
            q_p = T_goal[p_i]
            path.append(q_p)
            p_i = P_goal[p_i]

    return np.array(path)

if __name__ == '__main__':
    map_struct = loadmap("../maps/map2.txt")
    print(map_struct.obstacles)
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    # Assuming points is an nx3 NumPy array
    points = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 7], [3.33, 2, 4]])  # Example points

    # Calculate the distance matrix
    #distance_matrix = np.sqrt(((points[:, np.newaxis] - points)**2).sum(axis=2))
    # Get the upper triangular part of the matrix, excluding the diagonal
    #upper_triangle_distances = np.triu(distance_matrix, k=1)
    #print(upper_triangle_distances)
    # If you need to get these distances as a flat array:
    #upper_triangle_flat = upper_triangle_distances[upper_triangle_distances != 0]
    #print(upper_triangle_flat)
    print(path)
