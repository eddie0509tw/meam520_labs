import numpy as np
from lib.calcJacobian import calcJacobian



def IK_velocity(q_in, v_in, omega_in):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :return:
    dq - 1 x 7 vector corresponding to the joint velocities. If v_in and omega_in
         are infeasible, then dq should minimize the least squares error. If v_in
         and omega_in have multiple solutions, then you should select the solution
         that minimizes the l2 norm of dq
    """

    ## STUDENT CODE GOES HERE

    dq = np.zeros((1, 7))

    v_in = v_in.reshape((3,1))
    omega_in = omega_in.reshape((3,1))
    J=calcJacobian(q_in)
    velocity = np.concatenate((v_in, omega_in))

    nan_indices = np.isnan(velocity).flatten()

    J[nan_indices] = 0

    velocity[nan_indices] = 0
    
    dq, _, _, _ = np.linalg.lstsq(J, velocity, rcond=None)

    return dq

if __name__ == '__main__':
    # Example usage
    q_current = np.array([1, 2, 0, 4, 2, 2, 0])
    v_desired = np.array([1, 0, 3])  # Example with NaN value in v_in
    omega_desired = np.array([np.nan,np.nan,np.nan])  # Example with NaN value in omega_in

    dq_solution = IK_velocity(q_current, v_desired, omega_desired)
    print(dq_solution)
