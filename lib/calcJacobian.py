import numpy as np
from lib.calculateFK import FK

def calcJacobian(q_in):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """

    J = np.zeros((6, 7))

    ## STUDENT CODE GOES HERE
    fk = FK()
    joint_positions, T0e = fk.forward(q_in)
    axis_of_rot = fk.get_axis_of_rotation(q_in)

    o_end =joint_positions[7]
    for i in range(7):
        z_ = axis_of_rot[: ,i]
        J[:3, i]=np.cross(z_,(o_end-joint_positions[i]))


    J[3: , :] = axis_of_rot

    return J

if __name__ == '__main__':
    q= np.array([0, 0, 0, 0, 0, np.pi, np.pi/4])
    print(np.round(calcJacobian(q),3))
