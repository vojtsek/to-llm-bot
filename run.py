import argparse
import pickle
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pynvml import *
from datasets import load_dataset

from model import FewShotPromptedLLM
from prompts import RESPONSE_PROMPTS, STATE_PROMPTS
from database import MultiWOZDatabase
from utils import parse_state, ExampleRetriever, ExampleFormatter, print_gpu_utilization

DOMAINS = [
    	'restaurant',
    	'hotel',
    	'attraction',
    	'train',
    	'taxi',
    	'police',
    	'hospital'
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="/home/hudecek/hudecek-troja/test-llm/hf_cache")
    parser.add_argument("--model_name", type=str, default="allenai/tk-instruct-3b-def-pos-neg-expl")
    parser.add_argument("--faiss_db", type=str, default="multiwoz-context-db.vec")
    parser.add_argument("--num_examples", type=int, default=2)
    parser.add_argument("--database_path", type=str, default="multiwoz_database")
    parser.add_argument("--hf_dataset", type=str, default="multi_woz_v22")
    parser.add_argument("--context_size", type=int, default=3)
    parser.add_argument("--ontology", type=str, default="ontology.json")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name,
                                                  low_cpu_mem_usage=True,
                                                  cache_dir=args.cache_dir,
                                                  device_map="auto",
                                                  load_in_8bit=True)
    model = FewShotPromptedLLM(model, tokenizer)
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
    for dialog in dataset['test']:
        if n > 500:
            break
        if len(dialog['services']) != 1:
            continue
        # if dialog['services'][0] not in ['restaurant']:
        #     continue
        n += 1
        history = []
        domain = dialog['services'][0]
        get_state_prompt = STATE_PROMPTS[domain]
        response_prompt = RESPONSE_PROMPTS[domain]
        dialogue_id = dialog['dialogue_id'].split('.')[0].lower()
        results[dialogue_id] = []
        results_wo_state[dialogue_id] = []
        for tn in range(0, len(dialog['turns']['utterance']), 2):
            question = dialog['turns']['utterance'][tn]
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
            print(question)
            retrieve_history = history + ["Customer: " + question]


            retrieved_examples = example_retriever.retrieve("\n".join(retrieve_history[-args.context_size:]),
                                                            k=args.num_examples)
            positive_examples = example_formatter.format(retrieved_examples,
                                                        input_keys=["context"],
                                                        output_keys=["state"])
            negative_examples = example_formatter.format(retrieved_examples,
                                                        input_keys=["context"],
                                                        output_keys=["state"],
                                                        corrupt_state=True)
            response_examples = example_formatter.format(retrieved_examples,
                                                        input_keys=["context", "state", "database"],
                                                        output_keys=["response"])
            domain = retrieved_examples[0]['domain']
            
            state = model(get_state_prompt,
                      positive_examples=positive_examples,
                      negative_examples=negative_examples,
                      history="\n".join(history),
                      utterance=question.strip())
            # print("Raw state:", state)
            parsed_state = parse_state(state, default_domain=domain)
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
            # final_state = new_gt_state
            print(final_state)
            
            database_results = database.query(domain=domain,
                                              constraints=final_state)
            database_results = {domain: len(database_results)}
            
            print("-" * 100)
            # print(f"Possitive response examples: {response_examples}")
            response = model(response_prompt,
                            positive_examples=response_examples,
                            negative_examples=[],
                            history="\n".join(history),
                            utterance=question.strip(),
                            state=json.dumps(final_state).replace("{", '[').replace("}", ']'),
                            database=str(database_results))
            print(response)
            print("=" * 100)
            history.append("Customer: " + question)
            history.append("Assistant: " + response)
            
            results[dialogue_id].append({
                "response": response,
                "state": final_state,
                # "active_domains": [topic]
            })
            results_wo_state[dialogue_id].append({
                "response": response,
            })

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    with open("results_wo_state.json", "w") as f:
        json.dump(results_wo_state, f, indent=4)