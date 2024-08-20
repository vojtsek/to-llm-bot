#! /bin/bash
#
# Usage: ./run_intereact.sh <model>
#
# supported models: gpt-3.5-turbo
#                   allenai/tk-instruct-11b-def-pos-neg-expl


# XXX add your conda/virtualenv/CUDA environment etc. setup here

# XXX make sure there's a file with your OpenAI key in it
ROOT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export OPENAI_API_KEY=`cat "$ROOT_DIR/openai_key.txt"`


MODEL=${1}
shift
echo python3 ${ROOT_DIR}/interact.py --model_name $MODEL --faiss_db ${ROOT_DIR}/mwoz_db.pkl --num_examples 2 --database_path ${ROOT_DIR}/multiwoz_database --context_size 2 --dataset multiwoz --ontology ${ROOT_DIR}/ontology.json --run_name `whoami` --goal_data ${ROOT_DIR}/goals_trial.json $@
