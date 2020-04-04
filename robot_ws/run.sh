echo "Please source your mara setup and python virtual env too."
echo " e.g. source v-python_env/bin/activate"
echo " e.g. source /home/username/ros2learn/environments/gym-gazebo2/provision/mara_setup.sh"


source install/setup.bash
ros2 run recycler_package recycler
