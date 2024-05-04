import sys
import numpy as np
from copy import deepcopy
from math import pi
from time import perf_counter

from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray
from scipy.spatial.transform import Rotation as R

import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

from lib.calculateFK import FK
from lib.calcJacobian import FK
from lib.IK_position_null import IK

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds

def my_ik(ik, target, seed):
    start = perf_counter()
    
    q, rollout, success,_ = ik.inverse(target, seed,method='J_pseudo', alpha=.5)
    stop = perf_counter()
    time_taken = stop - start

    if success:
        print("Solution found in {time:2.2f} seconds ({it} iterations).".format(time=time_taken, it=len(rollout)))
        # arm.safe_move_to_position(q)
    else:
        print('IK Failed for this target using this seed.')

    return q  # return the q as a backup for further use

def move_around(team, detector):

    if team == 'red':
        moving_positions = [np.array([-0.07204 , 0.02167, -0.08044, -1.54437 , 0.00174,  1.56596 , 0.63294]),
                            np.array([-0.1163 , -0.15663, -0.04539 ,-1.8518 , -0.00713 , 1.69533 , 0.62515]),
                            np.array([ -0.17192, -0.09386 ,-0.14729, -1.7474 ,  0.02442 , 1.65245,  0.48828])]
    else:
        moving_positions = [np.array([0.14423, -0.05794,  0.19823, -1.63671 , 0.01141,  1.5799 ,  1.12744]),
                            np.array([0.13482 ,-0.07781,  0.20605 ,-1.76212,  0.01601 , 1.68593 , 1.12383]),
                            np.array([0.46276497, -0.1099812, -0.15097874, -1.77047204, 0.01556398, 1.68251959, 1.11978701])]

    for i in range(len(moving_positions)):
        arm.safe_move_to_position(moving_positions[i])
        new_detections_list = detector.get_detections()
        if len(new_detections_list) != 0:
            print("Got new detections")
            break
        
    return new_detections_list

def pick_place_static_block(team, start_position, arm, detector, ik, fk, subsequent_scanning=False, blk_cnt=0, num_static=4):

    if team == 'red':
        # Starting configuration to view the blocks
        start_position = np.array([-0.01779206-pi/9, -0.76012354+pi/6,  0.01978261, -2.34205014+pi/8, 0.02984053, 1.54119353+pi/11, 0.75344866]) # Can change this to b emore closer to the blocks
    else:
        start_position = np.array([-0.01779206+pi/9, -0.76012354+pi/6,  0.01978261, -2.34205014+pi/8, 0.02984053, 1.54119353+pi/11, 0.75344866])

    arm.safe_move_to_position(start_position) # on your mark!

    _, T0e = fk.forward(start_position)

    # get the transform from camera to panda_end_effector
    H_ee_camera = detector.get_H_ee_camera()

    if subsequent_scanning == True:
        for i in range(num_static):
            # Get the block transform in base frame
            detections_list = detector.get_detections()
            if len(detections_list) == 0:
                detections_list = move_around(team, detector)

            name, pose  = detections_list[0]
            target = T0e @ H_ee_camera @ pose
            print(name,'\n',target)

            # Transform the axes so that Z points downwards
            here = transform_axes(target)
            here[2,-1] = 0.23              # Setting the Z position


            # Solve IK
            q_found = my_ik(ik, here, start_position)
            print(here)
            print(q_found)
            

            arm.open_gripper()
            arm.safe_move_to_position(q_found)
            arm.exec_gripper_cmd(0.045, 30)   # arm.close_gripper()

            drop_pos, lift_pos = place_static_block(team, blk_number=blk_cnt)

            arm.safe_move_to_position(start_position)
            arm.safe_move_to_position(drop_pos)
            if check_if_gripped(arm):
                blk_cnt += 1

            arm.open_gripper()
            arm.safe_move_to_position(lift_pos)
            arm.safe_move_to_position(start_position)

        print("Block counter : ", blk_cnt)
    else:
        ik_q = []

        # Detect some blocks...
        arm.open_gripper()
        for (name, pose) in detector.get_detections():
            # Get the block transform in base frame
            target = T0e @ H_ee_camera @ pose
            print(name,'\n',target)

            # Transform the axes so that Z points downwards
            here = transform_axes(target)
            here[2,-1] = 0.235              # Setting the Z position

            # Solve IK
            q_found = my_ik(ik, here, start_position)

            # Append ik solutions for all blocks
            ik_q.append(q_found)

            print()
            print("-----------------------------------------------------")


        for i in range(len(ik_q)):
            arm.open_gripper()
            arm.safe_move_to_position(ik_q[i])
            arm.exec_gripper_cmd(0.045, 50)   # arm.close_gripper()
            drop_pos, lift_pos = place_static_block(team, blk_number=blk_cnt)

            arm.safe_move_to_position(start_position)
            arm.safe_move_to_position(drop_pos)
            arm.open_gripper()
            arm.safe_move_to_position(lift_pos)
            arm.safe_move_to_position(start_position)

    return blk_cnt
    
def Rot_x(theta):
    return np.array([[1,0,0],[0,np.cos(theta), -np.sin(theta)],[0,np.sin(theta), np.cos(theta)]])

def Rot_y(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]])

def Rot_z(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0],[0,0,1]])

def transform_axes(mat):
    up_vec = mat[:3,:3].T @ np.array([0,0,1])
    idx = int(np.where(np.abs(np.round(up_vec,1)) == 1)[0])
    sign_ = int(np.round(up_vec,1)[idx] / 1)
    
    if sign_ > 0 and idx == 0:
        int_rot = Rot_y(pi/2) @ Rot_x(pi)
    elif sign_ < 0 and idx == 0:
        int_rot = Rot_y(-pi/2) @ Rot_x(pi)
    elif sign_ > 0 and idx == 1:
        int_rot = Rot_x(-pi/2) @ Rot_x(pi)
    elif sign_ < 0 and idx == 1:
        int_rot = Rot_x(pi/2) @ Rot_x(pi)
    elif sign_ > 0 and idx == 2:
        int_rot = Rot_x(pi)
    else: 
        int_rot = np.eye(3)

    
    R_new = np.eye(4)
    R_new[:3, -1] = mat[:3, -1]
    R_new[:3,:3] = mat[:3,:3] @ int_rot
    if R_new[0,0] < 0:
        R_new[:3,:3] = R_new[:3,:3] @ Rot_z(pi)
    return R_new

def place_static_block(team, blk_number):
    if team == 'red':
        # Working seeds for red dropping positions
        
        # For the location - target[:3,-1] = np.array([0.531, 0.2, 0.285+n(0.05)]) , target[:3,:3] = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
        lifting_pos = [np.array([0.23518,  0.0994,   0.1286,  -2.06014, -0.01529,  2.15867,  1.15704]),
                        np.array([0.20799,  0.05161 , 0.15541, -1.98724, -0.00894 , 2.03821 , 1.15263]),
                        np.array([0.19175 , 0.02157 , 0.17061, -1.8944,  -0.00389 , 1.91565,  1.14904]),
                        np.array([0.18587,  0.0104,   0.17591, -1.78018, -0.00186,  1.79041,  1.14758]),
                        np.array([ 0.19097,  0.01971 , 0.172 ,  -1.64135, -0.00339,  1.66077,  1.14865]),
                        np.array([0.20957,  0.05276,  0.15756, -1.47092, -0.00828 , 1.52303 , 1.15192]),
                        np.array([0.24721,  0.11779 , 0.12712 ,-1.25206 ,-0.01521 , 1.36893 , 1.15581]),
                        np.array([0.31447,  0.2464 ,  0.06321, -0.92447, -0.01673,  1.17044 , 1.15465])]

        # For the location - target[:3,-1] = np.array([0.531, 0.2, 0.24+n(0.05)]) , target[:3,:3] = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
        dropping_seeds = [np.array([0.27003,  0.15671,  0.09306 ,-2.10913, -0.01887,  2.26508 , 1.15943]),
                        np.array([0.23194,  0.09385 , 0.13186, -2.05373, -0.01469 , 2.14672 , 1.15663]),
                        np.array([0.20589,  0.04779 , 0.15742, -1.97887, -0.00834 , 2.02605 , 1.1522]),
                        np.array([0.1907 ,  0.01958 , 0.17157 ,-1.88397, -0.00354 , 1.90327,  1.14879]),
                        np.array([0.18587 , 0.01037 , 0.17594, -1.76748, -0.00185 , 1.77769 , 1.14757]),
                        np.array([ 0.19216 , 0.02187 , 0.17108 ,-1.6259 , -0.00373,  1.64744,  1.14889]),
                        np.array([0.21235,  0.05761,  0.15539, -1.45165, -0.00893,  1.50857,  1.15233]),
                        np.array([0.25238,  0.12677,  0.12273, -1.2261,  -0.01586 , 1.35196,  1.15609])]
    else:
        # Working seeds for blue dropping positions

        # For the location - target[:3,-1] = np.array([0.531, -0.2, 0.285+n(0.05)]) , target[:3,:3] = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
        lifting_pos = [np.array([-0.16935,  0.10048, -0.19601, -2.06011,  0.02348,  2.15855,  0.40798]),
                        np.array([-0.17532 , 0.0519 , -0.18864, -1.98724 , 0.0109 ,  2.0382 ,  0.41677]),
                        np.array([-0.17866 , 0.02162,-0.18383 ,-1.8944 ,  0.0042 ,  1.91565 , 0.42153]),
                        np.array([-0.17981,  0.01041, -0.18201 ,-1.78018 , 0.00193 , 1.79041 , 0.42317]),
                        np.array([-0.17981 , 0.01975 ,-0.18331 ,-1.64135 , 0.00361 , 1.66077,  0.42199]),
                        np.array([-0.18101 , 0.05302, -0.18734, -1.47092 , 0.00988,  1.52302 , 0.41778]),
                        np.array([-0.18965,  0.11889, -0.19166, -1.25203,  0.02307 , 1.36881 , 0.41004]),
                        np.array([-0.22817 , 0.24907, -0.18129, -0.92424 , 0.04829 , 1.16981 , 0.40027])]
        
        # For the location - target[:3,-1] = np.array([0.531, -0.2, 0.24+n(0.05)]) , target[:3,:3] = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
 
        dropping_seeds = [np.array([-0.16184 , 0.15929, -0.204 ,  -2.10899 , 0.04181,  2.26461,  0.39533]),
                        np.array([-0.17007 , 0.09482 ,-0.19518 ,-2.0537 ,  0.02189 , 2.14662 , 0.40908]),
                        np.array([-0.17577 , 0.04804, -0.18804 ,-1.97886 , 0.00999,  2.02604 , 0.41741]),
                        np.array([-0.17886 , 0.01963 ,-0.18351, -1.88397 , 0.00379 , 1.90327 , 0.42183]),
                        np.array([-0.17984 , 0.01038, -0.18199 ,-1.76748,  0.00192 , 1.77769 , 0.42318]),
                        np.array([-0.17982,  0.02191 ,-0.1836,  -1.6259,   0.00401 , 1.64744,  0.42171]),
                        np.array([-0.18137 , 0.05792, -0.18784, -1.45165,  0.01083 , 1.50855 , 0.41717]),
                        np.array([-0.19141,  0.12801 ,-0.19188 ,-1.22606,  0.02494 , 1.35181,  0.40906])]

    return dropping_seeds[blk_number], lifting_pos[blk_number]

def check_if_gripped(arm):
    gripper_state = arm.get_gripper_state()
    gripper_position = gripper_state['position']
    gripper_dist = abs(gripper_position[1] + gripper_position[0])
    print("gripper_dist = {}".format(gripper_dist))

    flag = False
    # CHECK ON HARDWARE #################################
    if gripper_dist >= 0.056 or gripper_dist <= 0.03:  # didn't catch & not exceed -> keep trying
        print("didn't catch...")
        flag = False
    elif 0.056 > gripper_dist > 0.03:
        print("Caught")
        flag = True

    return flag

def place_dynamic_block(team, blk_number):
    if team == 'red':
        dropping_seeds = [np.array([-0.23931,  0.42277,  0.23466, -2.1494  , 1.4977 ,  1.63713, -1.77287]),
                          np.array([-0.19126,  0.30861,  0.18166, -2.17094 , 1.52267,  1.60277, -1.68812]),
                          np.array([-0.14032,  0.20717,  0.12556, -2.17656 , 1.54043,  1.57748, -1.59642]),
                          np.array([-0.08989,  0.11948,  0.07091, -2.1661  , 1.55165,  1.56161, -1.49989]),
                          np.array([-0.04303,  0.04606,  0.02157, -2.13956 , 1.55759,  1.55382, -1.40031]),
                          np.array([-0.00174, -0.01274, -0.0202 , -2.09707 , 1.55979,  1.55181, -1.29904]),
                          np.array([0.03243 , -0.05663, -0.05315, -2.03878 , 1.55979,  1.55309, -1.1969 ]),
                          np.array([0.05829 , -0.08529, -0.07683, -1.96464 , 1.55902,  1.55539, -1.09424])]
        
        lifting_seeds = [np.array([-0.19626,  0.31947,  0.18717, -2.1695 ,  1.5205 ,  1.60582, -1.69694]),
                         np.array([-0.14545,  0.21671,  0.13119, -2.17672,  1.53896,  1.57958, -1.60584]),
                         np.array([-0.09482,  0.12761,  0.07619, -2.16787,  1.55079,  1.5628 , -1.50971]),
                         np.array([-0.04742,  0.05275,  0.02612, -2.14293,  1.55719,  1.55429, -1.41036]),
                         np.array([-0.00558, -0.00753, -0.0164 , -2.10203,  1.5597 ,  1.55183, -1.30921]),
                         np.array([ 0.02936, -0.05292, -0.05026, -2.04532,  1.55985,  1.55288, -1.20714]),
                         np.array([ 0.05612, -0.08311, -0.07489, -1.97278,  1.55909,  1.55517, -1.10453]),
                         np.array([ 0.0732 , -0.09777, -0.08982, -1.88403,  1.55878,  1.55686, -1.00127])]
    else:
        dropping_seeds = [np.array([-0.95963,  0.81312,  0.32555, -1.5207 ,  0.89798,  1.18198, -1.56877]),
                          np.array([-0.94515,  0.72259,  0.29906, -1.5363 ,  0.95598,  1.12991, -1.51930]),
                          np.array([-0.9293 ,  0.64385,  0.27136, -1.53462,  1.01562,  1.07691, -1.45863]),
                          np.array([-0.91464,  0.57781,  0.24581, -1.51638,  1.07727,  1.02616, -1.38609]),
                          np.array([-0.90412,  0.5253 ,  0.22661, -1.48175,  1.14127,  0.98054, -1.30120]),
                          np.array([-0.90098,  0.48726,  0.21875, -1.43022,  1.2075 ,  0.94276, -1.20374]),
                          np.array([-0.90866,  0.46515,  0.22839, -1.36024,  1.27448,  0.9157 , -1.09348]),
                          np.array([-0.93069,  0.46186,  0.26409, -1.26844,  1.33789,  0.90301, -0.96994])]
        
        lifting_seeds = [np.array([-0.94669,  0.73112 , 0.30181, -1.53553,  0.95011,  1.13521, -1.52474]),
                         np.array([-0.93088,  0.65117 , 0.2741 , -1.53554,  1.00957,  1.08216, -1.46522]),
                         np.array([-0.91597,  0.58382 , 0.24815, -1.51894,  1.071  ,  1.03105, -1.39389]),
                         np.array([-0.90489,  0.52992 , 0.22811, -1.48596,  1.13477,  0.98479, -1.31026]),
                         np.array([-0.90087,  0.49037 , 0.21886, -1.43616,  1.2008 ,  0.94611, -1.21406]),
                         np.array([-0.9073 ,  0.46658 , 0.22642, -1.36814,  1.26784,  0.91783, -1.10509]),
                         np.array([-0.92774,  0.46121 , 0.25901, -1.27876,  1.33195,  0.90349, -0.98293]),
                         np.array([-0.9653 ,  0.48011 , 0.32987, -1.16141,  1.3831 ,  0.90846, -0.84608])]

    return dropping_seeds[blk_number], lifting_seeds[blk_number]

def pick_place_dynamic_block(team, arm, intermediate_position, dynamic_start_time, blk_counter):
    arm.open_gripper()
    if team == 'red':
        pos_2 = np.array([ 0.49479,  0.0505,   0.41046, -1.82479, -0.02109,  1.87104,  1.69642])  
        start_position_dyn_table = np.array([ 0.53284 , 1.0967 ,  0.75091 ,-1.22765,  0.7214 ,  1.65682 ,-1.25596])     # Seed to get into position just outside the table
        sweeping_pos = np.array([0.81009,  1.02481,  0.8289,  -1.45305 , 0.85513,  1.99263, -1.31996])
        picked_seed = np.array([0.37053, -0.32285,  0.60391, -2.20561,  2.16784,  2.37738, -1.31926])
    else:
        pos_2 = np.array([-0.46324,  0.05124, -0.44283, -1.82478,  0.02297,  1.87101, -0.12697])
        start_position_dyn_table = np.array([0.40660178, -1.72556938, -1.52057465, -0.95552095, -0.26955849, 1.56790008, -0.77764917]) # Seed with lower height outside table
        sweeping_pos = np.array([0.82591, -1.71208, -1.45934, -0.64209, -0.32318,  1.38184, -0.8676])
        picked_seed = np.array([-1.39347 , 1.23789 , 0.78353, -0.55751,  0.71615 , 0.56591+0.05 ,-0.92937])

    arm.safe_move_to_position(pos_2)
    arm.safe_move_to_position(start_position_dyn_table)

    is_gripped = False  # suppose it didn't catch
    
    while is_gripped is False:

        if perf_counter() - dynamic_start_time <= 60:
            print("time is ok")

            gripper_state = arm.get_gripper_state()
            gripper_position = gripper_state['position']
            # print("gripper_position = {}".format(gripper_position))
            gripper_dist = abs(gripper_position[1] + gripper_position[0])
            print("gripper_dist = {}".format(gripper_dist))

            arm.safe_move_to_position(sweeping_pos)

            if gripper_dist >= 0.05 or gripper_dist <= 0.03:  # didn't catch & not exceed -> keep trying
                print("didn't catch...")
                arm.open_gripper()

                start_time = perf_counter()
                while perf_counter() - start_time < 4.5:  # whatever
                    # print("wait to grip again...")
                    pass

                arm.exec_gripper_cmd(0.045, 30)

                is_gripped = False

            elif 0.05 > gripper_dist > 0.03:
                print("caught!")

                is_gripped = True
                isSucceed = True
                arm.safe_move_to_position(picked_seed)

                drop_pos, lift_pos = place_dynamic_block(team, blk_counter)

                arm.safe_move_to_position(drop_pos)
                if check_if_gripped(arm):
                    blk_counter += 1

                arm.open_gripper()
                arm.safe_move_to_position(lift_pos)

        elif perf_counter() - dynamic_start_time > 60:  # didn't catch & exceed -> stop and back to neutral pos
            print("time exceeds!")

            is_gripped = True  # jump out of the loop cuz it has already exceeded the time!
            isSucceed = False

            gripper_state = arm.get_gripper_state()
            gripper_position = gripper_state['position']
            # print("gripper_position = {}".format(gripper_position))
            gripper_dist = abs(gripper_position[1] + gripper_position[0])
            print("gripper_dist = {}".format(gripper_dist))

            if 0.05 > gripper_dist > 0.03:
                print("exceeded but caught!")
                is_gripped = True
                isSucceed = True

    if isSucceed is False:
        print("failed...")
        arm.open_gripper()

        return blk_counter, isSucceed

    print("succeeded!")

    return blk_counter, isSucceed

if __name__ == "__main__":

    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")
    arm = ArmController()
    detector = ObjectDetector()
    ik = IK()
    fk = FK()

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n") # go!

    # joint_pos = arm.get_positions()
    # print("Joint pos : ", joint_pos)
    # STUDENT CODE HERE
    if team == 'red':
        start_position = np.array([-0.01779206-pi/9, -0.76012354+pi/6,  0.01978261, -2.34205014+pi/8, 0.02984053, 1.54119353+pi/11, 0.75344866]) # Can change this to be more closer to the blocks
        intermediate_position  =np.array([ 0.39137,  0.05391 , 0.33123, -1.02744 , 0.0046 ,  1.19294,  0.78261])
    else:
        start_position = np.array([-0.01779206+pi/9, -0.76012354+pi/6,  0.01978261, -2.34205014+pi/8, 0.02984053, 1.54119353+pi/11, 0.75344866]) # Can change this to be more closer to the blocks
        intermediate_position= np.array([-0.01779206-(2*pi)/9, -0.76012354+pi/6,  0.01978261, -2.34205014+pi/8, 0.02984053, 1.54119353+pi/11, 0.75344866])

    glob_blk_cnt = 0

    # Static block stacking
    glob_blk_cnt = pick_place_static_block(team, start_position, arm, detector, ik, fk, subsequent_scanning=True, blk_cnt=glob_blk_cnt, num_static=4)

    print("Number of blocks done : ", glob_blk_cnt)

    arm.safe_move_to_position(intermediate_position)
    
    print("Going for dynamic blocks")
    # Dynamic block 
    NUM_DYNAMIC_BLKS = 2
    for j in range(NUM_DYNAMIC_BLKS):
        dynamic_start_time = perf_counter()
        glob_blk_cnt, _ = pick_place_dynamic_block(team, arm, intermediate_position, dynamic_start_time, blk_counter=glob_blk_cnt)

    # Move around...

    # # END STUDENT CODE
