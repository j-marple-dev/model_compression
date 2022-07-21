#!/bin/bash
#
# Docker build image, run container, execute last container.
#
# - Author: Jongkuk Lim
# - Contact: limjk@jmarple.ai

xhost +

ORG=jmarpledev

PRJ_NAME=${PWD##*/}
PRJ_NAME=${PRJ_NAME,,}

DOCKER_TAG=$ORG/$PRJ_NAME

CMD_ARGS=( ${@} )
CMD_ARGS=${CMD_ARGS[*]:1}

if [[ $2 == :* ]]; then
    DOCKER_TAG=$DOCKER_TAG$2
    CMD_ARGS=${CMD_ARGS[*]:2}
fi

if [ "$1" = "build" ]; then
    echo "Building a docker image with tagname $DOCKER_TAG and arguments $CMD_ARGS"
    docker build . -t $DOCKER_TAG $CMD_ARGS --build-arg UID=`id -u` --build-arg GID=`id -g`
elif [ "$1" = "run" ]; then
    echo "Run a docker image with tagname $DOCKER_TAG and arguments $CMD_ARGS"

    docker run -tid --privileged --gpus all \
        -e DISPLAY=${DISPLAY} \
        -e TERM=xterm-256color \
        -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
        -v /dev:/dev \
        -v $PWD:/home/user/$PRJ_NAME \
        --network host \
        $CMD_ARGS \
        $DOCKER_TAG /bin/bash

    last_cont_id=$(docker ps -qn 1)
    echo $(docker ps -qn 1) > $PWD/.last_exec_cont_id.txt

    docker exec -ti $last_cont_id /bin/bash
elif [ "$1" = "exec" ]; then
    echo "Execute the last docker container"

    last_cont_id=$(tail -1 $PWD/.last_exec_cont_id.txt)
    docker start ${last_cont_id}
    docker exec -ti ${last_cont_id} /bin/bash
else
    echo ""
    echo "============= $0 [Usages] ============"
    echo "1) $0 build : build docker image"
    echo "      build --no-cache : Build docker image without cache"
    echo "2) $0 run : launch a new docker container"
    echo "3) $0 exec : execute last container launched"
fi

