from typing import Any, Dict
import random
import json

from langchain.vectorstores import VectorStore

from prompts import FewShotPrompt, SimpleTemplatePrompt


class ExampleRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, text: str, k: int = 2, corrupt: bool = False, output_key: str = 'state') -> list[Dict]:
        result = self.vector_store.similarity_search(text, k=k)
        examples = [{'input': doc.page_content,
                     'output': doc.metadata[output_key]}
                     for doc in result]
        if output_key == 'state':
            if corrupt:
                examples = [self._corrupt_examples(example) for example in examples]
            for example in examples:
                example['output'] = json.dumps(example['output']).replace('{','[').replace('}', ']')
        return examples
    
    def _corrupt_examples(self, example: Dict) -> Dict:
        for key, val in example['output'].items():
            example['output'][key] = random.choice(['chinese', 'expensive', '5', 'free', 'italian', 'cheap', '10', 'indian'])
        return example


class SimplePromptedLLM:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, prompt: SimpleTemplatePrompt, **kwargs: Any):
        filled_prompt = prompt(**kwargs)
        return self._predict(filled_prompt)

    def _predict(self, text):
        input_ids = self.tokenizer.encode(text,return_tensors="pt").to(self.model.device)
        output = self.model.generate(input_ids, max_length=55)
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return output


class FewShotPromptedLLM(SimplePromptedLLM):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)

    def __call__(self, prompt: FewShotPrompt, positive_examples: list[Dict], negative_examples: list[Dict], **kwargs: Any):
        filled_prompt = prompt(positive_examples, negative_examples, **kwargs)
        return self._predict(filled_prompt)
