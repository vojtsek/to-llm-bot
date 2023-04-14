import argparse
import pickle
import glob
import os

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.docstore.document import Document

from definitions import MW_FEW_SHOT_DOMAIN_DEFINITIONS, SGD_FEW_SHOT_DOMAIN_DEFINITIONS
from loaders import load_mwoz, load_sgd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_faiss_db')
    parser.add_argument('--model', default='sentence-transformers/all-mpnet-base-v2', help='Embedding model name; text-embedding-ada-002 for OpenAI, sentence-transformers/all-mpnet-base-v2 for HuggingFace')
    parser.add_argument('--database_path', default='multiwoz_database')
    parser.add_argument('--context_size', type=int, default=3)
    parser.add_argument('--embeddings', default='huggingface', help='huggingface or openai')
    parser.add_argument('--dataset', default='multiwoz')
    parser.add_argument('--total', default=500, type=int)
    parser.add_argument('--split', default='train', type=str)
    args = parser.parse_args()

    if args.embeddings == 'huggingface':
        embeddings = HuggingFaceEmbeddings(model_name=args.model)
    else:
        embeddings = OpenAIEmbeddings(document_model_name=args.model,
                                      query_model_name=args.model,
                                      openai_api_key=os.environ.get('OPENAI_API_KEY', ''))

    if args.dataset == 'multiwoz':
        available_domains = [d for d in MW_FEW_SHOT_DOMAIN_DEFINITIONS.keys() if d != 'bus']
        data_gen = load_mwoz(args.database_path, args.context_size, split=args.split, total=args.total, available_domains=available_domains, shuffle=True, only_single_domain=True)
    else:
        available_domains = list(SGD_FEW_SHOT_DOMAIN_DEFINITIONS.keys())
        data_gen = load_sgd(args.context_size, split=args.split, total=args.total, available_domains=available_domains, shuffle=True)
    docs = []
    for turn in data_gen:
        doc = Document(page_content=turn['page_content'],
                       metadata=turn['metadata'])
        docs.append(doc)
    faiss_vs = FAISS.from_documents(documents=docs,
                                    embedding=embeddings)
    with open(args.output_faiss_db, 'wb') as f:
        pickle.dump(faiss_vs, f)
