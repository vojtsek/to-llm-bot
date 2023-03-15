#! /bin/bash
set -xe
base_conda_env="/home/hudecek/miniconda3/etc/profile.d/conda.sh"

printf "\nSourcing base conda environment $base_conda_env \n"
source "$base_conda_env"


conda_env="llm-env"
printf "Activating conda environment $conda_env\n\n"
conda activate $conda_env

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/11.7/lib64/;
export OPENAI_API_KEY=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
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
MODEL=${1:=gpt-3.5-turbo}
shift
#python run.py --model_name $MODEL --faiss_db multiwoz-context-state-update.vec --num_examples 2 --database_path multiwoz_database --hf_dataset multi_woz_v22 --context_size 3 --output results.txt
python run.py --model_name $MODEL --faiss_db multiwoz-state-update-ctx2.vec --num_examples 2 --database_path multiwoz_database --hf_dataset multi_woz_v22 --context_size 2 --output results.txt $@
