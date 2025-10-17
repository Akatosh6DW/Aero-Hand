import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'aero_hand_open_description'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mohit Yadav',
    maintainer_email='mohityadav@tetheria.ai',
    description='This package contains the URDF description of the Aero Hand Open robotic hand along with necessary launch and RViz configuration files to visualize it in RViz.',
    license='CC-BY-SA-4.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)