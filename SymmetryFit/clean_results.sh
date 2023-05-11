#!/bin/bash

while true; do

read -p "Do you want to delete all the results in results/? (yes/no) " yn

case $yn in 
	yes ) echo ok, we will proceed;
		break;;
	no ) echo exiting...;
		exit;;
	* ) echo invalid response;;
esac

done

echo Deleting all results...

rm -rf results
