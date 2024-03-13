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
            "purePursuit_ros = my_waypoint_follower.purePursuit_ros:main",
            "Opt_PurePursuit =  my_waypoint_follower.Opt_purePursuit_ros:main",
            "MPC_ros = my_waypoint_follower.MPC_ros:main",
            "MPC_ros_tuning = my_waypoint_follower.MPC_ros_tuning:main",
            "Opt_MPC = my_waypoint_follower.Opt_MPC_ros:main",
            "MPCC_ros = my_waypoint_follower.MPCC_ros:main",
            "Opt_MPCC = my_waypoint_follower.Opt_MPCC_ros:main"
        ],
    },
)
