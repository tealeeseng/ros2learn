#!/bin/bash

plist=`ps -ef | grep ros2 | awk '{print $2}'`
for pid in $plist
do
    echo 'kill process' $pid
    kill $pid
done


