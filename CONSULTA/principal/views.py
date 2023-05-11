from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
import json

from deep_translator import GoogleTranslator

from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
import torch
import time

model = AutoModelForCausalLM.from_pretrained(
    "databricks/dolly-v2-3b",
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b")

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

def home(request):
    args = {}
    return render(request,"home.html",args)

def search(request):
    user = request.user
    if request.is_ajax() and request.method == 'GET':
        text = request.GET['query']
        query = GoogleTranslator(source='pt', target='en').translate(text)
        # result = query
        answer = prompt(query)

        result = GoogleTranslator(source='en', target='pt').translate(answer)

        return HttpResponse(json.dumps({'result':result}), content_type="application/json")
