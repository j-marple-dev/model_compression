#!/bin/bash
#
# Shell script for formating, linting and unit test
#
# - Author: Jongkuk Lim
# - Contact: limjk@jmarple.ai

# Bash 3 does not support hash dictionary.
# hput and hget are alternative workaround.
# Usage)
# hput $VAR_NAME $KEY $VALUE
hput() {
    eval "$1""$2"='$3'
}

# Usage)
# `hget $VAR_NAME $KEY`
hget() {
    eval echo '${'"$1$2"'#hash}'
}

# Define command names
CMD_NAME=(
    "format"
    "lint"
    "test"
    "doc"
    "doc_server"
    "init_conda"
    "init_precommit"
    "init"
    "all"
    )

# Define descriptions
hput CMD_DESC format "Run formating"
hput CMD_DESC lint "Run linting check"
hput CMD_DESC test "Run unit test"
hput CMD_DESC doc "Generate MKDocs document"
hput CMD_DESC doc_server "Run MKDocs hosting server (in local)"
hput CMD_DESC init_conda "Create conda environment with default name"
hput CMD_DESC init_precommit "Install pre-commit plugin"
hput CMD_DESC init "Run init-conda and init-precommit"
hput CMD_DESC all "Run formating, linting and unit test"

# Define commands
hput CMD_LIST format "black . && isort . && docformatter -i -r . --wrap-summaries 88 --wrap-descriptions 88"
hput CMD_LIST lint "env PYTHONPATH=. pytest --pylint --mypy --flake8 --ignore tests --ignore cpp --ignore config --ignore save"
hput CMD_LIST test "env PYTHONPATH=. pytest tests --cov=scripts --cov-report term-missing --cov-report html"
hput CMD_LIST doc "env PYTHONPATH=. mkdocs build --no-directory-urls"
hput CMD_LIST doc_server "env PYTHONPATH=. mkdocs serve -a 127.0.0.1:8000 --no-livereload"
hput CMD_LIST init_conda "conda env create -f environment.yml"
hput CMD_LIST init_precommit "pre-commit install --hook-type pre-commit --hook-type pre-push"
hput CMD_LIST init "`hget CMD_LIST init_conda` && `hget CMD_LIST init_precommit`"
hput CMD_LIST all "`hget CMD_LIST format` && `hget CMD_LIST lint` && `hget CMD_LIST test`"

for _arg in $@
do
    if [[ `hget CMD_LIST $_arg` == "" ]]; then
        echo "$_arg is not valid option!"
        echo "--------------- $0 Usage ---------------"
        for _key in ${CMD_NAME[@]}
        do
            echo "$0 $_key - `hget CMD_DESC $_key`"
        done
        exit 0
    else
        cmd=`hget CMD_LIST $_arg`
        echo "Run $cmd"
        eval $cmd

        result=$?
        if [ $result -ne 0 ]; then
            exitCode=$result
        fi
    fi
done

exit $exitCode
