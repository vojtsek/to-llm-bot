import sys
sys.path = [p for p in sys.path if 'schmidtova' not in p and 'mukherjee' not in p]
print(sys.path)
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
import transformers
import random

from model import (
    FewShotPromptedLLM,
    SimplePromptedLLM,
    FewShotOpenAILLM,
    ZeroShotOpenAILLM,
    FewShotOpenAIChatLLM,
    ZeroShotOpenAIChatLLM,
    FewShotAlpaca,
    ZeroShotAlpaca
    )
from loaders import load_mwoz, load_sgd
from delex import prepareSlotValuesIndependent, delexicalise, delexicaliseReferenceNumber
from definitions import MW_FEW_SHOT_DOMAIN_DEFINITIONS, MW_ZERO_SHOT_DOMAIN_DEFINITIONS, SGD_FEW_SHOT_DOMAIN_DEFINITIONS, SGD_ZERO_SHOT_DOMAIN_DEFINITIONS, multiwoz_domain_prompt, sgd_domain_prompt

from database import MultiWOZDatabase
from utils import parse_state, ExampleRetriever, ExampleFormatter, print_gpu_utilization, SGDEvaluator
from mwzeval.metrics import Evaluator as MWEvaluator


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

transformers.set_seed(42)


def lexicalize(results, domain, response):
    if domain not in results:
        return response
    elif len(results[domain]) == 0:
        return response
    item = results[domain][0]
    extend_dct = {f"{domain}_{key}": val for key, val in item.items()}
    item.update(extend_dct)
    item.update({f"value_{key}": val for key, val in item.items()})
    item["choice"] = str(len(results[domain]))
    for key, val in item.items():
        x = f"[{key}]"
        if x in response:
            response = response.replace(x, val)
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="/home/hudecek/hudecek/hf_cache")
    parser.add_argument("--model_name", type=str, default="allenai/tk-instruct-11b-def-pos-neg-expl")
    parser.add_argument("--faiss_db", type=str, default="multiwoz-context-db.vec")
    parser.add_argument("--num_examples", type=int, default=2)
    parser.add_argument("--dials_total", type=int, default=100)
    parser.add_argument("--database_path", type=str, default="multiwoz_database")
    parser.add_argument("--dataset", type=str, default="multiwoz")
    parser.add_argument("--context_size", type=int, default=3)
    parser.add_argument("--ontology", type=str, default="ontology.json")
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--use_gt_state", action='store_true')
    parser.add_argument("--use_gt_domain", action='store_true')
    parser.add_argument("--use_zero_shot", action='store_true')
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--goal_data", type=str)
    args = parser.parse_args()
    config = {
        "model_name": args.model_name,
        "faiss_db": args.faiss_db,
        "num_examples": args.num_examples,
        "dataset": args.dataset,
        "context_size": args.context_size,
        "use_gt_state": args.use_gt_state,
        "use_zero_shot": args.use_zero_shot,
        "use_gt_domain": args.use_gt_domain,
    }
    if 'tk-instruct-3b' in args.model_name:
        model_name = 'tk-3B'
    elif 'tk-instruct-11b' in args.model_name:
        model_name = 'tk-11B'
    elif 'opt-iml-1.3b' in args.model_name:
        model_name = 'opt-iml-1.3b'
    elif 'opt-iml-30b' in args.model_name:
        model_name = 'opt-iml-30b'
    elif 'NeoXT' in args.model_name:
        model_name = 'GPT-NeoXT-20b'
    elif 'gpt-3.5' in args.model_name:
        model_name = 'ChatGPT'
    elif args.model_name == 'alpaca':
        model_name = 'Alpaca-LoRA'
    else:
        model_name = 'GPT3.5'
    if 'mukherjee' not in args.run_name:
        wandb.init(project='llmbot-interact', entity='metric', config=config, settings=wandb.Settings(start_method="fork"))
    else:
        wandb.init(project='llmbot-interact', entity='humaneai-diaser', config=config, settings=wandb.Settings(start_method="fork"))
    wandb.run.name = f'{args.run_name}-{args.dataset}-{model_name}-examples-{args.num_examples}-ctx-{args.context_size}'
    report_table = wandb.Table(columns=['id', 'goal', 'context', 'raw_state', 'parsed_state', 'response', 'predicted_domain'])

    mw_dial_goals = []
    with open(args.goal_data, "rt") as fd:
        data = json.load(fd)
        mw_dial_goals = [dial['goal']['message'] for did, dial in data.items()]
    if args.model_name.startswith("text-"):
        model_factory = ZeroShotOpenAILLM if args.use_zero_shot else FewShotOpenAILLM
        model = model_factory(args.model_name)
        domain_model = ZeroShotOpenAILLM(args.model_name)
    elif args.model_name.startswith("gpt-"):
        model_factory = ZeroShotOpenAIChatLLM if args.use_zero_shot else FewShotOpenAIChatLLM
        model = model_factory(args.model_name)
        domain_model = ZeroShotOpenAIChatLLM(args.model_name)
    elif any([n in args.model_name for n in ['opt', 'NeoXT']]):
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        model_w = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                       low_cpu_mem_usage=True,
                                                       cache_dir=args.cache_dir,
                                                       device_map="auto",
                                                       load_in_8bit=True)
        model_factory = SimplePromptedLLM if args.use_zero_shot else FewShotPromptedLLM
        model = model_factory(model_w, tokenizer, type="causal")
        domain_model = SimplePromptedLLM(model_w, tokenizer, type="causal")
    elif 'alpaca' in args.model_name:
        model_factory = ZeroShotAlpaca if args.use_zero_shot else FewShotAlpaca
        model = model_factory(model_name="Alpaca-LoRA")
        domain_model = ZeroShotAlpaca(model_name="Alpaca-LoRA")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        model_w = AutoModelForSeq2SeqLM.from_pretrained(args.model_name,
                                                    low_cpu_mem_usage=True,
                                                    cache_dir=args.cache_dir,
                                                    device_map="auto",
                                                    load_in_8bit=True)
        model_factory = SimplePromptedLLM if args.use_zero_shot else FewShotPromptedLLM
        model = model_factory(model_w, tokenizer, type="seq2seq")
        domain_model = SimplePromptedLLM(model_w, tokenizer, type="seq2seq")

    with open(args.faiss_db, 'rb') as f:
        faiss_vs = pickle.load(f)
    with open(args.ontology, 'r') as f:
        ontology = json.load(f)
    if args.dataset == 'multiwoz':
        domain_prompt = multiwoz_domain_prompt
        database = MultiWOZDatabase(args.database_path)
        state_vs = faiss_vs
        delex_dic = prepareSlotValuesIndependent(args.database_path)
    else:
        domain_prompt = sgd_domain_prompt
        state_vs = faiss_vs
        delex_dic = None
    example_retriever = ExampleRetriever(faiss_vs)
    state_retriever = ExampleRetriever(state_vs)
    example_formatter = ExampleFormatter(ontology=ontology)

    history = []
    last_dial_id = None
    total = args.dials_total
    dialogue_id = 1
    tn = 0
    total_state = {}
    goal = random.choice(mw_dial_goals)
    for msg in goal:
        print(msg)
    print(f'>>>>> Please, use id {wandb.run.id}-{dialogue_id} <<<<<')
    while True:
        user_input = input('User> ').lower()
        if '/end' in user_input:
            wandb.log({"examples": report_table})
            break
        if '/new' in user_input:
            dialogue_id += 1
            tn = 0
            history = []
            total_state = {}
            print('=' * 100)
            goal = random.choice(mw_dial_goals)
            for msg in goal:
                print(msg)
            print(f'>>>>> Please, use id {wandb.run.id}-{dialogue_id} <<<<<')
            previous_domain = None
            continue
        tn += 1
        question = user_input
        retrieve_history = history + ["Customer: " + question]
        retrieved_examples = example_retriever.retrieve("\n".join(retrieve_history[-args.context_size:]), k=20)
        retrieved_domains = [example['domain'] for example in retrieved_examples]
        selected_domain, dp = domain_model(domain_prompt, predict=True, history="\n".join(history[-2:]), utterance=F"Customer: {question.strip()}")
        if args.dataset == 'multiwoz':
            available_domains = list(MW_FEW_SHOT_DOMAIN_DEFINITIONS.keys())
        else:
            available_domains = list(SGD_FEW_SHOT_DOMAIN_DEFINITIONS.keys())
        if args.verbose:
            print(f"PREDICTED DOMAIN: {selected_domain}")
        if selected_domain not in available_domains:
            selected_domain = random.choice(available_domains)
        if args.dataset == 'multiwoz':
            domain_definition = MW_ZERO_SHOT_DOMAIN_DEFINITIONS[selected_domain] if args.use_zero_shot else MW_FEW_SHOT_DOMAIN_DEFINITIONS[selected_domain]
        else:
            domain_definition = SGD_ZERO_SHOT_DOMAIN_DEFINITIONS[selected_domain] if args.use_zero_shot else SGD_FEW_SHOT_DOMAIN_DEFINITIONS[selected_domain]
        retrieved_examples = [example for example in retrieved_examples if example['domain'] == selected_domain]
        num_examples = min(len(retrieved_examples), args.num_examples)
        num_state_examples = 5
        state_examples = [example for example in state_retriever.retrieve("\n".join(retrieve_history[-args.context_size:]), k=20) if example['domain'] == selected_domain][:num_state_examples]
        positive_state_examples = example_formatter.format(state_examples[:num_state_examples],
                                                           input_keys=["context"],
                                                           output_keys=["state"],
                                                           )
                                                           #use_json=True)
        negative_state_examples = example_formatter.format(state_examples[:num_state_examples],
                                                           input_keys=["context"],
                                                           output_keys=["state"],
                                                           corrupt_state=True)
        response_examples = example_formatter.format(retrieved_examples[:num_examples],
                                                     input_keys=["context", "full_state", "database"],
                                                     output_keys=["response"],
                                                     use_json=True)
        
        state_prompt = domain_definition.state_prompt
        response_prompt = domain_definition.response_prompt
        
        try:
            kwargs = {
                "history": "\n".join(history),
                "utterance": question.strip()
            }
            if not args.use_zero_shot:
                kwargs["positive_examples"] = positive_state_examples
                kwargs["negative_examples"] = [] # negative_state_examples
            state, filled_state_prompt = model(state_prompt, predict=True, **kwargs)
        except:
            state = "{}"

        parsed_state = parse_state(state, default_domain=selected_domain)
        if selected_domain not in parsed_state:
            parsed_state[selected_domain] = {}
        if not isinstance(parsed_state[selected_domain], dict):
            parsed_state[selected_domain] = {}
        keys_to_remove = [k for k in parsed_state[selected_domain].keys() if k not in domain_definition.expected_slots]
        for k in keys_to_remove:
            del parsed_state[selected_domain][k]
        try:
            for domain, ds in parsed_state.items():
                for slot, value in ds.items():
                    pass
        except:
            parsed_state = {selected_domain: {}}
        
        final_state = {}
        for domain, ds in parsed_state.items():
            if domain in available_domains:
                final_state[domain] = ds
        
        for domain, dbs in final_state.items():
            if domain not in total_state:
                total_state[domain] = dbs
            else:
                for slot, value in dbs.items():
                    value = str(value)
                    if value not in ['dontcare', 'none', '?', ''] and len(value) > 0:
                        total_state[domain][slot] = value
        
        print(f"Belief State: {total_state}", flush=True)

        if args.dataset == 'multiwoz':
            database_results = {domain: database.query(domain=domain, constraints=ds)
                                for domain, ds in total_state.items() if len(ds) > 0}
        else:
            database_results = turn['metadata']['database']
        logger.info(f"Database Results: {database_results}")
        print(f"Database Results: {database_results[selected_domain][0] if selected_domain in database_results and len(database_results[selected_domain]) > 0 else 'EMPTY'}", flush=True)
        
        try:
            kwargs = {
                "history": "\n".join(history),
                "utterance": question.strip(),
                "state": json.dumps(total_state), #.replace("{", '<').replace("}", '>'),
                "database": str({domain: len(results) for domain, results in database_results.items()})
            }
            if not args.use_zero_shot:
                kwargs["positive_examples"] = response_examples
                kwargs["negative_examples"] = []

            # response, filled_prompt = "IDK", "-"
            response, filled_prompt = model(response_prompt, predict=True, **kwargs)
        except:
            response = ''

        if args.dataset == 'multiwoz':
            response = delexicalise(response, delex_dic)
            response = delexicaliseReferenceNumber(response)
        
        print(f"Response: {response}", flush=True)
        print(f"Lexicalized response: {lexicalize(database_results, selected_domain, response)}", flush=True)

        history.append("Customer: " + question)
        report_table.add_data(f"dial_{dialogue_id}-turn_{tn}", ' '.join(goal), " ".join(history), state, json.dumps(final_state), response, selected_domain)
        history.append("Assistant: " + response)
        
