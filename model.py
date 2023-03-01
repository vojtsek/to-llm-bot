from typing import Any, Dict

from prompts import FewShotPrompt, SimpleTemplatePrompt


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
        # print(filled_prompt)
        return self._predict(filled_prompt)
