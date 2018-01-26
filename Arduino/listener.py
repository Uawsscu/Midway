#!/usr/bin/env phthon

import sys
import rospy
from std_msgs.msg import UInt8


def talker(msg):
    pub = rospy.Publisher('setLed', UInt8, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    pub.publish(int(msg))


if __name__ == '__main__':
    talker(sys.argv[1])