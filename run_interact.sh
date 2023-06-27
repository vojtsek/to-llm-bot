#! /bin/bash
set -xe
base_conda_env="/home/hudecek/miniconda3/etc/profile.d/conda.sh"
printf "\nSourcing base conda environment $base_conda_env \n"
source "$base_conda_env"

conda_env="llm-env"
printf "Activating conda environment $conda_env\n\n"
conda activate $conda_env

export LD_LIBRARY_PATH=/opt/cuda/11.7/lib64/;
export CUDA_HOME=
export OPENAI_API_KEY=`cat /home/hudecek/.openai-ufal`
ROOT_DIR="/lnet/troja/work/people/hudecek/test-llm"

if [ -d "$CUDA_DIR_OPT" ] ; then
  CUDA_DIR=$CUDA_DIR_OPT
  export CUDA_HOME=$CUDA_DIR
  export THEANO_FLAGS="cuda.root=$CUDA_HOME,device=gpu,floatX=float32"
  export PATH=$PATH:$CUDA_DIR/bin
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_DIR/cudnn/$CUDNN_version/lib64:$CUDA_DIR/lib64
  export CPATH=$CUDA_DIR/cudnn/$CUDNN_version/include:$CPATH
fi

# gpt-3.5-turbo
# allenai/tk-instruct-11b-def-pos-neg-expl
env
MODEL=${1}
shift
which python3
echo $PYTHONPATH
python3 -c 'import sys; print(sys.path)'
python3 ${ROOT_DIR}/interact.py --model_name $MODEL --faiss_db ${ROOT_DIR}/mw-context-2-20perdomain.vec --num_examples 2 --database_path ${ROOT_DIR}/multiwoz_database --context_size 2 --dataset multiwoz --ontology ${ROOT_DIR}/ontology.json --run_name `whoami` --goal_data /home/hudecek/datasets/multiwoz2/test.json $@
