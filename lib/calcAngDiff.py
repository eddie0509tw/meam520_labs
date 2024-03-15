import numpy as np


def calcAngDiff(R_des, R_curr):
    """
    Helper function for the End Effector Orientation Task. Computes the axis of rotation
    from the current orientation to the target orientation

    This data can also be interpreted as an end effector velocity which will
    bring the end effector closer to the target orientation.

    INPUTS:
    R_des - 3x3 numpy array representing the desired orientation from
    end effector to world
    R_curr - 3x3 numpy array representing the "current" end effector orientation

    OUTPUTS:
    omega - 0x3 a 3-element numpy array containing the axis of the rotation from
    the current frame to the end effector frame. The magnitude of this vector
    must be sin(angle), where angle is the angle of rotation around this axis
    """
    omega = np.zeros(3)
    ## STUDENT CODE STARTS HERE
    R_rel = R_curr.T @ R_des
    S_rel = (R_rel - R_rel.T) / 2
    a = np.array([S_rel[2,1], S_rel[0, 2], S_rel[1, 0]])
    #if np.linalg.norm(a) == 0:
    #    eigenvalues, eigenvectors = np.linalg.eigh(R_rel)
    #    a = eigenvectors[:, np.isclose(eigenvalues, 1)]

    omega = R_rel @ a

    return omega

if __name__ == '__main__':
        R_des = np.eye(3)
        R_curr = np.eye(3)
        print(calcAngDiff(R_des, R_curr))
