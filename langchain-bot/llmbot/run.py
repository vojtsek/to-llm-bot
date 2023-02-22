import argparse
import pickle

from datasets import load_dataset
from llmbot.classifiers import VectorSimilarityTopicClassiffier
from llmbot.bots.multiwoz.multiwoz import MultiWOZBot
from llmbot.bots.banking import BankingBot

def bot_factory(bot_type, faiss_vs, bot_data_dir):
    if bot_type == 'multiwoz':
        return MultiWOZBot(faiss_vs, bot_data_dir)
    elif bot_type == 'banking':
        return BankingBot(faiss_vs)
    else:
        raise ValueError(f'Unknown bot type: {bot_type}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--faiss_db')
    parser.add_argument('--bot_type', choices=['multiwoz', 'banking'], dest='bot_type')
    parser.add_argument('--bot_data_dir', default='bots/')
    parser.add_argument('--hf_dataset', default=None, help='HuggingFace dataset name')
    args = parser.parse_args()
    with open(args.faiss_db, 'rb') as f:
        faiss_vs = pickle.load(f)
    topic_clf = VectorSimilarityTopicClassiffier(faiss_vs)
    bot = bot_factory(args.bot_type, topic_clf, args.bot_data_dir)
    if args.hf_dataset:
        dataset = load_dataset(args.hf_dataset)
        bot.run_with_dataset(dataset)
    else:
        bot.run()
