from manipulation.manipulationWQueue import *

if __name__ == '__main__':

  rospy.init_node("sp_demo1", anonymous = True)
  man = Manipulation()

  print("===============")
  print(" Arm Terminal  ")
  print("===============")

  while True:
    response = raw_input("To attempt to pick enter (y) to exit enter (n): ")
    if(response == "n"):
      exit(0)
    if(response == "y"):

      
      print("Moving arm")

      # get new joint angles updated by callback
      joint_data = man.joints
      # open the gripper to max
      man.setGripper(80)

      # move to object
      man.jointMove(joint_data)

      # grab object
      man.setGripper()

      # lift object a bit
      joint_data[1] = joint_data[1] - 10
      man.jointMove(joint_data)

      # put it back down
      joint_data[1] = joint_data[1] + 10
      man.jointMove(joint_data)

      # open gripper to release object
      man.setGripper(80)

      # return to home position
      man.jointMove()

