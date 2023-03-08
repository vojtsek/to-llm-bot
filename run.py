import argparse
import pickle
import json
import tqdm
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from pynvml import *
from datasets import load_dataset
import wandb
import logging

from model import FewShotPromptedLLM, OpenAILLM, OpenAIChatLLM
from prompts import DOMAIN_DEFINITIONS
from database import MultiWOZDatabase
from utils import parse_state, ExampleRetriever, ExampleFormatter, print_gpu_utilization
from mwzeval.metrics import Evaluator

DOMAINS = [
    	'restaurant',
    	'hotel',
    	'attraction',
    	'train',
    	'taxi',
    	'police',
    	'hospital'
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="/home/hudecek/hudecek/hf_cache")
    parser.add_argument("--model_name", type=str, default="allenai/tk-instruct-3b-def-pos-neg-expl")
    parser.add_argument("--faiss_db", type=str, default="multiwoz-context-db.vec")
    parser.add_argument("--num_examples", type=int, default=2)
    parser.add_argument("--database_path", type=str, default="multiwoz_database")
    parser.add_argument("--hf_dataset", type=str, default="multi_woz_v22")
    parser.add_argument("--context_size", type=int, default=3)
    parser.add_argument("--ontology", type=str, default="ontology.json")
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--run_name", type=str, default="")
    args = parser.parse_args()
    wandb.init(project='llmbot')
    if 'tk-instruct-3b' in args.model_name:
        model_name = 'tk-3B'
    elif 'tk-instruct-11b' in args.model_name:
        model_name = 'tk-11B'
    elif 'opt-iml-1.3b' in args.model_name:
        model_name = 'opt-iml-1.3b'
    elif 'opt-iml-30b' in args.model_name:
        model_name = 'opt-iml-30b'
    elif 'gpt' in args.model_name:
        model_name = 'ChatGPT'
    else:
        model_name = 'GPT3.5'
    wandb.run.name = f'{args.run_name}-{model_name}-examples-{args.num_examples}-ctx-{args.context_size}'
    report_table = wandb.Table(columns=['id', 'context', 'raw_state', 'parsed_state', 'response'])
    if args.model_name.startswith("text-"):
        model = OpenAILLM(args.model_name)
    elif args.model_name.startswith("gpt-"):
        model = OpenAIChatLLM(args.model_name)
    elif 'opt' in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                    low_cpu_mem_usage=True,
                                                    cache_dir=args.cache_dir,
                                                    device_map="auto",
                                                    load_in_8bit=True)
        model = FewShotPromptedLLM(model, tokenizer, type="causal")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name,
                                                    low_cpu_mem_usage=True,
                                                    cache_dir=args.cache_dir,
                                                    device_map="auto",
                                                    load_in_8bit=True)
        model = FewShotPromptedLLM(model, tokenizer, type="seq2seq")

    database = MultiWOZDatabase(args.database_path)
    with open(args.faiss_db, 'rb') as f:
        faiss_vs = pickle.load(f)
    example_retriever = ExampleRetriever(faiss_vs)
    with open(args.ontology, 'r') as f:
        ontology = json.load(f)
    example_formatter = ExampleFormatter(ontology=ontology)

    history = []
    n = 1
    results = {}
    results_wo_state = {}
    dataset = load_dataset(args.hf_dataset)
    for n, dialog in enumerate(tqdm.tqdm(dataset['test'])):
        # if len(dialog['services']) != 1:
        #     continue
        # if dialog['services'][0] not in ['restaurant']:
        #     continue
        n += 1
        if n > 100:
            break
        history = []
        dialogue_id = dialog['dialogue_id'].split('.')[0].lower()
        results[dialogue_id] = []
        results_wo_state[dialogue_id] = []
        total_state = {}
        print('=' * 100)

        for tn in range(0, len(dialog['turns']['utterance']), 2):
            question = dialog['turns']['utterance'][tn]
            gold_response = dialog['turns']['utterance'][tn+1]
            gt_state = dialog['turns']['frames'][tn]['state']
            if len(gt_state) == 0:
                gt_state = {}
            else:
                gt_state = gt_state[0]['slots_values']
                gt_state = {k: v[0] for k, v in zip(gt_state['slots_values_name'], gt_state['slots_values_list']) }
            new_gt_state = {}
            for sl, val in gt_state.items():
                domain, name = sl.split('-')
                if domain not in new_gt_state:
                    new_gt_state[domain] = {name: val}
                else:
                    new_gt_state[domain][name] = val
            retrieve_history = history + ["Customer: " + question]
            retrieved_examples = example_retriever.retrieve("\n".join(retrieve_history[-args.context_size:]), k=5)
            retrieved_domains = [example['domain'] for example in retrieved_examples]
            selected_domain = Counter(retrieved_domains).most_common(1)[0][0]
            retrieved_examples = [example for example in retrieved_examples if example['domain'] == selected_domain]
            num_examples = min(len(retrieved_examples), args.num_examples)
            positive_examples = example_formatter.format(retrieved_examples[:num_examples],
                                                        input_keys=["context"],
                                                        output_keys=["state"])
            negative_examples = example_formatter.format(retrieved_examples[:num_examples],
                                                        input_keys=["context"],
                                                        output_keys=["state"],
                                                        corrupt_state=True)
            response_examples = example_formatter.format(retrieved_examples[:num_examples],
                                                        input_keys=["context", "state", "database"],
                                                        output_keys=["response"])
            
            domain_definition = DOMAIN_DEFINITIONS[selected_domain]
            state_prompt = domain_definition.state_prompt
            response_prompt = domain_definition.response_prompt
            
            try:
                state = model(state_prompt,
                                positive_examples=positive_examples,
                                negative_examples=negative_examples,
                                history="\n".join(history),
                                utterance=question.strip())
            except:
                state = {}

            
            parsed_state = parse_state(state, default_domain=selected_domain)
            if selected_domain not in parsed_state:
                parsed_state[selected_domain] = {}
            keys_to_remove = [k for k in parsed_state[selected_domain].keys() if k not in domain_definition.expected_slots]
            for k in keys_to_remove:
                del parsed_state[selected_domain][k]
            try:
                for domain, ds in parsed_state.items():
                    for slot, value in ds.items():
                        pass
            except:
                parsed_state = {domain: {}}
            
            final_state = {}
            for domain, ds in parsed_state.items():
                if domain in DOMAINS:
                    final_state[domain] = ds
            
            for domain, dbs in final_state.items():
                if domain not in total_state:
                    total_state[domain] = dbs
                else:
                    for slot, value in dbs.items():
                        value = str(value)
                        if value not in ['dontcare', 'none', '?', ''] and len(value) > 0:
                            total_state[domain][slot] = value
            
            # final_state = new_gt_state
            print('-' * 100)
            print(f"Question: {question}", flush=True)
            print(f"Selected domain: {selected_domain}", flush=True)
            logger.info(f"Raw State: {state}")
            print(f"Raw State: {state}", flush=True)
            logger.info(f"Parsed State: {final_state}")
            print(f"Parsed State: {final_state}", flush=True)
            logger.info(f"Total State: {total_state}")
            print(f"Total State: {total_state}", flush=True)

            database_results = {domain: len(database.query(domain=domain, constraints=ds))
                                for domain, ds in total_state.items() if len(ds) > 0}
            logger.info(f"Database Results: {database_results}")
            print(f"Database Results: {database_results}", flush=True)
            
            try:
                response = model(response_prompt,
                                 positive_examples=response_examples,
                                 negative_examples=[],
                                 history="\n".join(history),
                                 utterance=question.strip(),
                                 state=json.dumps(total_state).replace("{", '<').replace("}", '>'),
                                 database=str(database_results))
            except:
                response = ''
            logger.info(f"Response: {response}")
            print(f"Response: {response}", flush=True)
            print(f"Gold Response: {gold_response}", flush=True)

            history.append("Customer: " + question)
            report_table.add_data(f"{dialogue_id}-{tn}", " ".join(history), state, json.dumps(final_state), response)
            history.append("Assistant: " + gold_response)
            
            results[dialogue_id].append({
                "response": response,
                "state": final_state,
            })
            results_wo_state[dialogue_id].append({
                "response": response,
            })
    wandb.log({"examples": report_table})

    evaluator = Evaluator(bleu=True, success=True, richness=True)
    eval_results = evaluator.evaluate(results)
    for metric, values in eval_results.items():
        if values is not None:
            for k, v in values.items():
                wandb.log({f"{k.ljust(15)}": v})

    evaluator = Evaluator(bleu=True, success=True, richness=True)
    eval_results = evaluator.evaluate(results_wo_state)
    for metric, values in eval_results.items():
        if values is not None:
            for k, v in values.items():
                wandb.log({f"GT_{k.ljust(15)}": v})
