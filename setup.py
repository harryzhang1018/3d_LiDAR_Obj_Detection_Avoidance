from setuptools import find_packages, setup

package_name = 'lidar_obstacle_detect_avoid'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='hzhang699@wisc.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lidar_ODA = lidar_obstacle_detect_avoid.lidar_ODA:main',
            'lidar_ODA_train = lidar_obstacle_detect_avoid.LiDAR_ODA_training:main'
        ],
    },
)
