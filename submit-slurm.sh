#! /bin/bash
set -xe
base_conda_env="/home/hudecek/miniconda3/etc/profile.d/conda.sh"

printf "\nSourcing base conda environment $base_conda_env \n"
source "$base_conda_env"


dataset="${1}"
shift
conda_env="${1}"
shift
printf "Activating conda environment $conda_env\n\n"
conda activate $conda_env

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/11.7/lib64/;
#export OPENAI_API_KEY=`cat ~/.openai-hudecek`
export OPENAI_API_KEY=`cat ~/.openai-ufal`
export HUGGINGFACEHUB_API_TOKEN=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

if [ -d "$CUDA_DIR_OPT" ] ; then
  CUDA_DIR=$CUDA_DIR_OPT
  export CUDA_HOME=$CUDA_DIR
  export THEANO_FLAGS="cuda.root=$CUDA_HOME,device=gpu,floatX=float32"
  export PATH=$PATH:$CUDA_DIR/bin
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_DIR/cudnn/$CUDNN_version/lib64:$CUDA_DIR/lib64
  export CPATH=$CUDA_DIR/cudnn/$CUDNN_version/include:$CPATH
fi
nvidia-smi
echo "$@"
# gpt-3.5-turbo
# text-davinci-003
# facebook/opt-iml-1.3b
# allenai/tk-instruct-3b-def-pos-neg-expl
# togethercomputer/GPT-NeoXT-Chat-Base-20B
MODEL=${1:=gpt-3.5-turbo}
shift
#python run.py --model_name $MODEL --faiss_db multiwoz-context-state-update.vec --num_examples 2 --database_path multiwoz_database --dataset multiwoz --context_size 3 --output results.txt
if [[ "${dataset}" == "sgd" ]]; then
    python run.py --model_name $MODEL --faiss_db sgd-context-2-10perdomain.vec --num_examples 2 --database_path multiwoz_database --context_size 2 --output results.txt --dataset sgd $@
else
    python run.py --model_name $MODEL --faiss_db mw-context-2-10perdomain.vec --num_examples 2 --database_path multiwoz_database --context_size 2 --output results.txt --dataset multiwoz $@
fi
