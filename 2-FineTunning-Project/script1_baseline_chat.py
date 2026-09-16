import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data"
EVAL_FILE = os.path.join(DATA_DIR, "eval_questions.jsonl")
BASE_MODEL = "microsoft/phi-1_5"

def load_eval_questions():
    questions = []
    with jsonlines.open(EVAL_FILE, "r") as reader:
        for obj in reader:
            questions.append(obj["question"])
    return questions

def chat(model, tokenizer, question):
    prompt = f"Questions: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Disable gradient calculation because we are only generating text (not training the model)
    with torch.no_grad():
        # Ask the model to generate a response
        output_ids = model.generate(
            **inputs,  # Pass all the tokenized input (input_ids, attention_mask, etc.) to the model
            max_new_tokens=80, # Generate at most 80 new tokens after the input prompt
            do_sample=False, # Disable random sampling so the model always picks the most likely next token This makes the output more deterministic and repeatable
            temperature=0.2, # Controls randomness in generation.
            pad_token_id=tokenizer.eos_token_id # Use the End-of-Sequence (EOS) token as the padding token
            # This prevents warnings if the model needs to pad sequences
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_token=True)
    return decoded


def run_baseline_evaluation():
    print(f"\nLoading base mdoel...(CPU may take some time)")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    questions = load_eval_questions()

    for q in questions:
        print(f"Question to the model -> {q}")
        answer = chat(model, tokenizer, q)
        print(f"Model's Answer: {answer}")
        print("-"*50)

if __name__=="__main__":
    run_baseline_evaluation()


"""
===========================================================
Padding Token vs EOS (End-of-Sequence) Token
===========================================================

LLMs expect every input in a batch to have the same length.

Example (max_length = 8):

"I love AI"
"I love learning Generative AI every day"

The first sentence is shorter, so we add padding tokens to make
both sequences the same length.

-----------------------------------------------------------
Padding Token (PAD)
-----------------------------------------------------------

PAD tokens are extra dummy tokens added to the end of shorter
sentences. They do not carry any meaning and are used only to
make all inputs the same length for batch processing.

Example:
"I love AI [PAD] [PAD] [PAD]"

-----------------------------------------------------------
EOS Token (End Of Sequence)
-----------------------------------------------------------

EOS marks the end of a sentence.

Example:
"I love AI [EOS]"

During text generation, the model keeps generating tokens until
it predicts the EOS token, which tells it to stop.

-----------------------------------------------------------
Why use EOS as the PAD token?
-----------------------------------------------------------

Many decoder-only LLMs (like Phi, Llama, GPT, and Mistral)
do not have a dedicated PAD token.

So we simply reuse the EOS token as the padding token:

tokenizer.pad_token = tokenizer.eos_token

or

pad_token_id = tokenizer.eos_token_id

Although the padding positions contain the EOS token, the model
ignores them using the attention mask, so they are treated only
as padding—not as the actual end of the sentence.

-----------------------------------------------------------
Summary
-----------------------------------------------------------

PAD  → Makes all input sequences the same length.
EOS  → Marks the end of a sentence and tells the model when to stop generating.
"""    