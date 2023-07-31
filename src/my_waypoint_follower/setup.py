from setuptools import setup

package_name = 'my_waypoint_follower'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yiminglinux',
    maintainer_email='22902716@sun.ac.za',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "pose_subscriber = my_waypoint_follower.pose_subscriber:main",
            "waypoint_follower = my_waypoint_follower.waypoint_follower:main",
            "MPC_ros = my_waypoint_follower.MPC_ros:main"
            "Opt_wayPoint_Follower =  my_waypoint_follower.Opt_wayPoint_Followermain."
        ],
    },
)
