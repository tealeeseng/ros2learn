from setuptools import setup

package_name = 'recycler_package'

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
    maintainer='leeseng',
    maintainer_email='tealeeseng@gmail.com',
    description='Recycler Robot poc package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'recycler = recycler_package.recycler:main'
        ],
    },
)
