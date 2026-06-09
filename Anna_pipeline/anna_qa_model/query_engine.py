import torch
from transformers import (AutoTokenizer,AutoModelForCausalLM)

class QueryEngine:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("abdullaho99/arna_model_tl_0.1429472407427701")
        self.tokenizer = AutoTokenizer.from_pretrained("abdullaho99/arna_model_tl_0.1429472407427701")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def search(self, query):
        prompt = f"Question: {query}\nAnswer:"
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=256
        )

        # Encode stop sequences
        stop_strings = ["\nQuestion:", "\n\n", "Question:"]
        stop_ids = [
            self.tokenizer.encode(s, add_special_tokens=False)
            for s in stop_strings
        ]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=60,  # safety ceiling only
                do_sample=False,
                repetition_penalty=1.3,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=[
                    self.tokenizer.eos_token_id,
                    self.tokenizer.encode("\n", add_special_tokens=False)[0],  # stop at newline
                ],
            )

        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the answer portion
        response = full.split("Answer:")[-1].strip()

        # Cut off at any next "Question:" if model keeps going
        if "Question:" in response:
            response = response.split("Question:")[0].strip()

        return response


if __name__=="__main__":
    query = QueryEngine()
    print(query.search("What are the tech stacks which ARNA Industry use?"))
