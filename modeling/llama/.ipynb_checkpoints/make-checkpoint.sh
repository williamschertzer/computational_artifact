#!/bin/bash
[[ -n $MKWD ]] || export MKWD=$(echo $(command cd $(dirname '$0'); pwd))

## GLOBAL VARIABLES
## -----------------------------------------------------------------------------

CONDAENV=llama-env

if [[ $(basename "${0}" 2> /dev/null) != "make.sh" ]]; then
    if ! conda activate $CONDAENV; then
        echo "Setting up $CONDAENV conda environment."
        conda create --solver=classic -n $CONDAENV python=3.12 numpy -c conda-forge || return 10
        conda activate $CONDAENV || return 11

        # Install packages.
	conda install jupyter
	conda install ipykernel
	python -m ipykernel install --user --display-name "Python (llama)"
        conda install cxx-compiler==1.5.2 cmake openmpi-mpicxx fftw # gcc 11
        conda install pytorch==2.2.2 pytorch-cuda cuda-toolkit -c pytorch -c nvidia
    fi

    export HF_HOME=/data/wschertzer/aem_aging/modeling/llama

    alias cdmk="cd $MKWD/"
    alias mk="$MKWD/make.sh"
    echo "Environment set up. You can now use 'mk' to execute this script."

    return 0
fi

jlab() {
    if ! grep -q "password" ~/.jupyter/jupyter_server_config.json; then
        echo "Please setup jupyter password."
        jupyter lab password
    fi
    port=${1:-8808}
    jupyter lab --port=$port --ip=0.0.0.0
}

## EXECUTE OR SHOW USAGE.
## -----------------------------------------------------------------------------
if [[ "$#" -lt 1 ]]; then
    echo -e "\nUSAGE:  mk <command> [options ...]"
    echo -e "\tSource this script to setup the terminal environment."
    echo -e "\nAvailable commands:"
    echo -e "------------------------------------------------------------------"

    echo -e "jlab               - Run Jupyter lab server."
    echo

else
    cd $MKWD
    "$@"
fi
