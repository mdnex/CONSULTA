from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
import torch
import time

model = AutoModelForCausalLM.from_pretrained(
    "databricks/dolly-v2-12b",
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b")

def prompt(s):
    global tokenizer
    global model

    input_ids = tokenizer(s, return_tensors="pt").input_ids.to('cuda')
    start = time.time()
    gen_tokens = model.generate(
      input_ids,
      do_sample=True,
      temperature=0.9,
      max_length=100
    )

    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    end = time.time()
    #print("time to generate: ", end - start)
    return gen_text
  
prompt("How do I write a book?")
