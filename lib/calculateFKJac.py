import numpy as np
from math import pi

class FK_Jac():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab 1 and 4 handout

        self.a = [0, 0, 0, 0.0825, 0.0825, 0, 0.088, 0]
        self.alpha = [0, -pi/2, pi/2, pi/2, pi/2, -pi/2, pi/2, 0]
        self.d = [0.141, 0.192, 0, 0.316, 0, 0.384, 0, 0.21]
        self.theta = None
        self.init_angle = np.array([ 0,  0,    0,     0, -pi/2,     0, pi/2, pi/4 ])

    def construct_theta(self, q):
        return [0, q[0], q[1],  q[2], q[3]+pi/2, q[4], q[5]-pi/2, q[6]]

    def build_DH(self, a, alpha, d, theta):
        A = np.array([[np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
                    [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
                    [0, np.sin(alpha), np.cos(alpha), d],
                    [0, 0, 0, 1]])
        return A

    def forward_expanded(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -10 x 3 matrix, where each row corresponds to a physical or virtual joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 10 x 4 x 4 homogeneous transformation matrix,
                  representing the each joint/end effector frame expressed in the
                  world frame
        """

        # Your code starts here

        jointPositions = np.zeros((10,3))
        T0e_ = np.zeros((10,4,4))
        T0e = np.identity(4)

        # Your code ends here
        # this is the offset wrt to that joint frame dn
        offset = np.array([  [0.0, 0.0 ,0.0],
                             [0.0, 0.0 ,0.0],
                             [0.0, 0.0, 0.195],
                             [0.0, 0.0, 0.0 ],
                             [0.0, 0.0, 0.125],
                             [0.0, 0.0, -0.015],
                             [0.0, 0.0, 0.051],
                             [0.0, 0.0, 0.0]])

        vj_offset = np.array([[0 ,0.1,-0.105],
                              [0 ,-0.1,-0.105]])

        self.theta = self.construct_theta(q) - self.init_angle
        for i, (a, alpha, d, theta) in enumerate(zip(self.a, self.alpha, self.d, self.theta)):
            A = self.build_DH(a, alpha, d, theta)
            T0e = T0e @ A
            R = T0e[:-1, :-1]
            if i == 7:
                for vj_off in vj_offset:
                    jointPositions[i] = T0e[:-1, -1] +  R @ vj_off
                    T0e_[i] = T0e
                    T0e_[i, :-1, -1] = jointPositions[i]
                    i+=1
                jointPositions[-1] = T0e[:-1, -1] +  R @ offset[-1]
                T0e_[-1] = T0e
                T0e_[-1, :-1, -1] = jointPositions[-1]
                break
            else:
                jointPositions[i] = T0e[:-1, -1] +  R @ offset[i]
                T0e_[i] = T0e
                T0e_[i, :-1, -1] = jointPositions[i]

        return jointPositions, T0e_

    # feel free to define additional helper methods to modularize your solution for lab 1
    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x9 np array of unit vectors describing the axis of rotation for each joint in the
                                 world frame

        """
        # STUDENT CODE HERE: This is a function needed by lab 2
        axis_of_rot = np.zeros((9,3))
        axis_of_rot[0, 2] = 1
        # Your code ends here
        _ , T0e = self.forward_expanded(q)
        #R_all = T0e[1: , :-1, :-1]
        print(T0e)
        for i in range(1, 9):
            R = T0e[i , :-1, :-1]
            axis_of_rot[i] = R @ np.array([0, 0, 1])

        #axis_of_rot[7] = R @ np.array([0, 0, 1])
        #axis_of_rot[8] = R @ np.array([0, 0, 1])
        return axis_of_rot.T

    def calcJacobian(self, q_in):
        """
        Calculate the full Jacobian of the end effector in a given configuration
        :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
        :return: J - 6 x 9 matrix representing the Jacobian, where the first three
        rows correspond to the linear velocity and the last three rows correspond to
        the angular velocity, expressed in world frame coordinates
        """

        J = np.zeros((6, 9))

        ## STUDENT CODE GOES HERE
        joint_positions, T0e = self.forward_expanded(q_in)
        axis_of_rot = self.get_axis_of_rotation(q_in)
        print(axis_of_rot)

        o_end =joint_positions[-1]
        for i in range(7):
            z_ = axis_of_rot[: ,i]
            J[:3, i]=np.cross(z_,(o_end-joint_positions[i]))
            J[3: , i] = z_


        for i in range(8, 10):
            z_ = axis_of_rot[: ,6]
            J[:3, i-1]=np.cross(z_,(o_end-joint_positions[i]))
            J[3: , i-1] = z_
        #J[3: , :] = axis_of_rot
        exit()

        return J

    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        # STUDENT CODE HERE

        return()

if __name__ == "__main__":

    fk = FK_Jac()

    # matches figure in the handout
    q = np.array([pi/5,pi/6,pi/3,-pi/2,0,pi/3,pi/4])

    joint_positions, T0e = fk.forward_expanded(q)

    #print("Joint Positions:\n",joint_positions)
    #print("End Effector Pose:\n",T0e)
    print(np.round(fk.calcJacobian(q),3))
