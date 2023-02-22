import argparse
import pickle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pynvml import *

from model import FewShotPromptedLLM, ExampleRetriever
from prompts import FewShotPrompt

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(1)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="/home/hudecek/hudecek-troja/test-llm/hf_cache")
    parser.add_argument("--model_name", type=str, default="allenai/tk-instruct-3b-def-pos-neg-expl")
    parser.add_argument("--faiss_db", type=str, default="multiwoz-context-db.vec")
    parser.add_argument("--num_examples", type=int, default=2)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name,
                                                  low_cpu_mem_usage=True,
                                                  cache_dir=args.cache_dir,
                                                  device_map="auto",
                                                  load_in_8bit=True)
    model = FewShotPromptedLLM(model, tokenizer)
    with open(args.faiss_db, 'rb') as f:
        faiss_vs = pickle.load(f)
    example_retriever = ExampleRetriever(faiss_vs)
    get_state_prompt = FewShotPrompt(template="""
Definition: Provide summary of the conversation in JSON with keys: area, food, pricerange.
Fill the values with what is mentioned in the text or leave empty but mention all three values
{}{}Now complete the following example:
input:{}
Customer: {}
output:""",
                             args_order=["history", "utterance"])
    response_prompt = FewShotPrompt(template="""
Definition: You are an assistant that helps people to book a restaurant.
The customer can ask for a restaurant by name, area, food, or price.
Provide final answer on separate line
If there is 0 restaurants in the database, ask the customer to change the request.
If you find a restaurant, provide [restaurant_name].
{}{}Now complete the following example:
input:{}
Customer: {}
output:""",
                             args_order=["history", "utterance"])
    history = []
    while True:
        inp = input(">")
        if 'exit' in inp:
            break
        context = "\n".join(history) + "\nCustomer: " + inp
        retrieved_positive = example_retriever.retrieve(context, k=args.num_examples, corrupt=False)
        retrieved_negative = example_retriever.retrieve(context, k=args.num_examples, corrupt=True)
        print(f"Possitive examples: {retrieved_positive}")
        print(f"Negative examples: {retrieved_negative}")

        state = model(get_state_prompt,
                      positive_examples=retrieved_positive,
                      negative_examples=retrieved_negative,
                      history="\n".join(history),
                      utterance=inp.strip())
        print(state)
        retrieved_positive = example_retriever.retrieve(inp, k=args.num_examples, corrupt=False, output_key="response")
        print(f"Possitive response examples: {retrieved_positive}")
        response = model(response_prompt,
                         positive_examples=retrieved_positive,
                         negative_examples=[],
                         history="\n".join(history),
                         utterance=inp.strip())
        print(response)
        history.append("Customer: " + inp)
        history.append("Assistant: " + response)
    print_gpu_utilization()
