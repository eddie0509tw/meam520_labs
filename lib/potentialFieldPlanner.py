import numpy as np
from math import pi, acos
from scipy.linalg import null_space
from copy import deepcopy
from lib.calculateFKJac import FK_Jac
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap


class PotentialFieldPlanner:

    # JOINT LIMITS
    lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint
    fk = FK_Jac()

    def __init__(self, tol=1e-4, max_steps=500, min_step_size=1e-5):
        """
        Constructs a potential field planner with solver parameters.

        PARAMETERS:
        tol - the maximum distance between two joint sets
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # YOU MAY NEED TO CHANGE THESE PARAMETERS

        # solver parameters
        self.tol = tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size
        #self.map = loadmap("../maps/map1.txt")


    ######################
    ## Helper Functions ##
    ######################
    # The following functions are provided to you to help you to better structure your code
    # You don't necessarily have to use them. You can also edit them to fit your own situation

    @staticmethod
    def attractive_force(target, current):
        """
        Helper function for computing the attactive force between the current position and
        the target position for one joint. Computes the attractive force vector between the
        target joint position and the current joint position

        INPUTS:
        target - 3x1 numpy array representing the desired joint position in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame

        OUTPUTS:
        att_f - 3x1 numpy array representing the force vector that pulls the joint
        from the current position to the target position
        """

        ## STUDENT CODE STARTS HERE
        psi = 1.0

        att_f = np.zeros((3, 1))
        f = -(current - target)
        f_n = np.linalg.norm(f)
        if f_n > 1.0:
            att_f = f / (np.linalg.norm(f))
        else:
            att_f = f
        ## END STUDENT CODE

        return att_f * psi

    @staticmethod
    def repulsive_force(obstacle, current, unitvec=np.zeros((3,1))):
        """
        Helper function for computing the repulsive force between the current position
        of one joint and one obstacle. Computes the repulsive force vector between the
        obstacle and the current joint position

        INPUTS:
        obstacle - 1x6 numpy array representing the an obstacle box in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame
        unitvec - 3x1 numpy array representing the unit vector from the current joint position
        to the closest point on the obstacle box

        OUTPUTS:
        rep_f - 3x1 numpy array representing the force vector that pushes the joint
        from the obstacle
        """

        ## STUDENT CODE STARTS HERE
        eta = 1.0
        rho_0 = 0.15
        rep_f = np.zeros((3, 1))
        closest_dist, unitvec = PotentialFieldPlanner.dist_point2box(current.T, obstacle.flatten())
        closest_dist = np.linalg.norm(closest_dist)
        if (closest_dist > 0) & (closest_dist <= rho_0):
            rep_f = -eta * (1/closest_dist - 1/rho_0) * 1/np.square(closest_dist) * unitvec
        ## END STUDENT CODE
        return rep_f.reshape(3,1)

    @staticmethod
    def dist_point2box(p, box):
        """
        Helper function for the computation of repulsive forces. Computes the closest point
        on the box to a given point

        INPUTS:
        p - nx3 numpy array of points [x,y,z]
        box - 1x6 numpy array of minimum and maximum points of box

        OUTPUTS:
        dist - nx1 numpy array of distance between the points and the box
                dist > 0 point outside
                dist = 0 point is on or inside box
        unit - nx3 numpy array where each row is the corresponding unit vector
        from the point to the closest spot on the box
            norm(unit) = 1 point is outside the box
            norm(unit)= 0 point is on/inside the box

         Method from MultiRRomero
         @ https://stackoverflow.com/questions/5254838/
         calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
        """
        # THIS FUNCTION HAS BEEN FULLY IMPLEMENTED FOR YOU

        # Get box info
        boxMin = np.array([box[0], box[1], box[2]])
        boxMax = np.array([box[3], box[4], box[5]])
        boxCenter = boxMin*0.5 + boxMax*0.5
        p = np.array(p)

        # Get distance info from point to box boundary
        dx = np.amax(np.vstack([boxMin[0] - p[:, 0], p[:, 0] - boxMax[0], np.zeros(p[:, 0].shape)]).T, 1)
        dy = np.amax(np.vstack([boxMin[1] - p[:, 1], p[:, 1] - boxMax[1], np.zeros(p[:, 1].shape)]).T, 1)
        dz = np.amax(np.vstack([boxMin[2] - p[:, 2], p[:, 2] - boxMax[2], np.zeros(p[:, 2].shape)]).T, 1)

        # convert to distance
        distances = np.vstack([dx, dy, dz]).T
        dist = np.linalg.norm(distances, axis=1)

        # Figure out the signs
        signs = np.sign(boxCenter-p)

        # Calculate unit vector and replace with
        unit = distances / dist[:, np.newaxis] * signs
        unit[np.isnan(unit)] = 0
        unit[np.isinf(unit)] = 0
        return dist, unit

    @staticmethod
    def compute_forces(target, obstacle, current):
        """
        Helper function for the computation of forces on every joints. Computes the sum
        of forces (attactive, repulsive) on each joint.

        INPUTS:
        target - 3x9 numpy array representing the desired joint/end effector positions
        in the world frame
        obstacle - nx6 numpy array representing the obstacle box min and max positions
        in the world frame
        current- 3x9 numpy array representing the current joint/end effector positions
        in the world frame

        OUTPUTS:
        joint_forces - 3x9 numpy array representing the force vectors on each
        joint/end effector
        """

        ## STUDENT CODE STARTS HERE
        n_o = obstacle.shape[0]
        joint_forces = np.zeros((3, 9))
        for i in range(joint_forces.shape[1]):
            rep_f = np.zeros((3, 1))
            for j in range(n_o):
                rep_f += PotentialFieldPlanner.repulsive_force(
                    obstacle[j].reshape(1, -1), current[:, i].reshape(-1, 1))
            att_f = PotentialFieldPlanner.attractive_force(
                    target[:, i].reshape(-1, 1), current[:, i].reshape(-1, 1))
            assert not np.any(np.isnan(att_f))
            assert not np.any(np.isnan(rep_f))
            joint_forces[:, i] = (att_f + rep_f).flatten()

        ## END STUDENT CODE

        return joint_forces

    @staticmethod
    def compute_torques(joint_forces, q):
        """
        Helper function for converting joint forces to joint torques. Computes the sum
        of torques on each joint.

        INPUTS:
        joint_forces - 3x9 numpy array representing the force vectors on each
        joint/end effector
        q - 1x7 numpy array representing the current joint angles

        OUTPUTS:
        joint_torques - 1x9 numpy array representing the torques on each joint
        """

        ## STUDENT CODE STARTS HERE
        joint_torques = np.zeros((1, 9))
        J = PotentialFieldPlanner.fk.calcJacobian(q) # 6x9
        Jv = J[:3] #3x9
        torque = np.zeros((9, 1))
        for i in range(joint_torques.shape[1]):
            J_ = np.zeros_like(Jv)
            J_[..., :i+1] = Jv[..., :i+1]
            f = joint_forces[..., i].reshape(-1, 1) #3x1
            torque += J_.T @ f

        joint_torques[:] = torque.flatten()
        ## END STUDENT CODE

        return joint_torques

    @staticmethod
    def q_distance(target, current):
        """
        Helper function which computes the distance between any two
        vectors.

        This data can be used to decide whether two joint sets can be
        considered equal within a certain tolerance.

        INPUTS:
        target - 1x7 numpy array representing some joint angles
        current - 1x7 numpy array representing some joint angles

        OUTPUTS:
        distance - the distance between the target and the current joint sets

        """

        ## STUDENT CODE STARTS HERE

        distance = np.linalg.norm(target-current)

        ## END STUDENT CODE

        return distance

    @staticmethod
    def compute_gradient(q, target, map_struct):
        """
        Computes the joint gradient step to move the current joint positions to the
        next set of joint positions which leads to a closer configuration to the goal
        configuration

        INPUTS:        print(rep_f.shape)
        q - 1x7 numpy array. the current joint configuration, a "best guess" so far for the final answer
        target - 1x7 numpy array containing the desired joint angles
        map_struct - a map struct containing the obstacle box min and max positions

        OUTPUTS:
        dq - 1x7 numpy array. a desired joint velocity to perform this task.
        """

        ## STUDENT CODE STARTS HERE

        dq = np.zeros((1, 7))
        target_pos, T0e_t = PotentialFieldPlanner.fk.forward_expanded(target)
        current_pos, T0e_c = PotentialFieldPlanner.fk.forward_expanded(q)
        forces = PotentialFieldPlanner.compute_forces(target_pos[1:].T, map_struct.obstacles, current_pos[1:].T)
        torques = PotentialFieldPlanner.compute_torques(forces, q)

        dq = torques/ np.linalg.norm(torques)
        ## END STUDENT CODE

        return dq[0, :7]

    ###############################
    ### Potential Feild Solver  ###
    ###############################

    def plan(self, map_struct, start, goal):
        """
        Uses potential field to move the Panda robot arm from the startng configuration to
        the goal configuration.

        INPUTS:
        map_struct - a map struct containing min and max positions of obstacle boxes
        start - 1x7 numpy array representing the starting joint angles for a configuration
        goal - 1x7 numpy array representing the desired joint angles for a configuration

        OUTPUTS:
        q - nx7 numpy array of joint angles [q0, q1, q2, q3, q4, q5, q6]. This should contain
        all the joint angles throughout the path of the planner. The first row of q should be
        the starting joint angles and the last row of q should be the goal joint angles.
        """

        q_path = np.array([]).reshape(0,7)
        cnt = 0
        q = start
        q_path = np.vstack((q_path,q))
        alpha = 3e-2
        last_q = np.zeros_like(q)
        while True:

            ## STUDENT CODE STARTS HERE

            # The following comments are hints to help you to implement the planner
            # You don't necessarily have to follow these steps to complete your code
            # Compute gradient
            dq = PotentialFieldPlanner.compute_gradient(q.flatten(), goal, map_struct)
            # TODO: this is how to change your joint angles
            print(dq)
            assert not np.any(np.isnan(dq))
            # Termination Conditions
            print(PotentialFieldPlanner.q_distance(goal, q))
            if PotentialFieldPlanner.q_distance(goal, q) < self.tol or cnt > self.max_steps: # TODO: check termination conditions
                break # exit the while loop if conditions are met!

            # YOU NEED TO CHECK FOR COLLISIONS WITH OBSTACLES
            # TODO: Figure out how to use the provided function
            q = q + alpha*dq
            q_path = np.vstack((q_path,q))
            # YOU MAY NEED TO DEAL WITH LOCAL MINIMA HERE
            # TODO: when detect a local minima, implement a random walk
            #if diff  <
            cnt+=1
            #last_q = q
            ## END STUDENT CODE

        return q_path

################################
## Simple Testing Environment ##
################################

if __name__ == "__main__":

    np.set_printoptions(suppress=True,precision=5)

    planner = PotentialFieldPlanner()

    # inputs
    map_struct = loadmap("../maps/map1.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])

    # potential field planning
    q_path = planner.plan(deepcopy(map_struct), deepcopy(start), deepcopy(goal))

    # show results
    for i in range(q_path.shape[0]):
        error = PotentialFieldPlanner.q_distance(q_path[i, :], goal)
        print('iteration:',i,' q =', q_path[i, :], ' error={error}'.format(error=error))

    print("q path: ", q_path)
