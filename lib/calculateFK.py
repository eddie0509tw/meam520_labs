import numpy as np
from math import pi

class FK():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout


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

    def forward(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -8 x 3 matrix, where each row corresponds to a rotational joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 4 x 4 homogeneous transformation matrix,
                  representing the end effector frame expressed in the
                  world frame
        """

        # Your Lab 1 code starts here

        T0e = np.identity(4)
        jointPositions = np.zeros((8,3))
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

        self.theta = self.construct_theta(q) - self.init_angle
        for i, (a, alpha, d, theta) in enumerate(zip(self.a, self.alpha, self.d, self.theta)):
            A = self.build_DH(a, alpha, d, theta)
            T0e = T0e @ A
            R = T0e[:-1, :-1]
            if i == 6:
                T_ = T0e
            jointPositions[i] = T0e[:-1, -1] +  R @ offset[i]

        return jointPositions, T_

    # feel free to define additional helper methods to modularize your solution for lab 1


    # This code is for Lab 2, you can ignore it ofr Lab 1
    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis of rotation for each joint in the
                                 world frame

        """
        # STUDENT CODE HERE: This is a function needed by lab 2
        axis_of_rot = np.zeros((7,3))
        axis_of_rot[0, 2] = 1
        # Your code ends here
        R_ = np.zeros((7,3,3))
        T0e = np.identity(4)

        self.theta = self.construct_theta(q) - self.init_angle
        for i, (a, alpha, d, theta) in enumerate(zip(self.a, self.alpha, self.d, self.theta)):
            A = self.build_DH(a, alpha, d, theta)
            T0e = T0e @ A
            R = T0e[:-1, :-1]
            if i >= 1:
                R_[i-1] = R

        for i in range(1, 7):
            axis_of_rot[i] = R_[i-1] @ np.array([0, 0, 1])

        return axis_of_rot.T

    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        return()

if __name__ == "__main__":

    fk = FK()

    # matches figure in the handout
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    #q = np.array([ 0,    0,     0, 0,  0, 0, 0 ])
    joint_positions, T0e = fk.forward(q)

    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",T0e)
