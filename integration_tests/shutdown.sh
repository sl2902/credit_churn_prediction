#!/usr/bin/env bash

function print_info {
    RESET="\e[0m"
    BOLD="\e[1m"
    YELLOW="\e[33m"
    echo -e "$YELLOW$BOLD [+] $1 $RESET"
}

echo "Clean up"
docker container stop $(docker container ls -q)
docker-compose down -v
docker-compose down --rmi=all
docker rmi -f image $(docker image ls -q)
unset EXPERIMENT_NAME