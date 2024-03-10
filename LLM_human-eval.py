import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from human_eval.data import write_jsonl, read_problems

import bitsandbytes




bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

access_token = "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"

model_name = "codellama/CodeLlama-7b-Python-hf"



tokenizer = AutoTokenizer.from_pretrained(model_name,token=access_token)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")

#device = "cuda:0"

#prompt = "from typing import List def has_close_elements(numbers: List[float], threshold: float) -> bool: """ 




#print(outputs)


def generate_one_completion(prompt) : #(problems[task_id]["prompt"])
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_ids_len = len(inputs["input_ids"][0])
    #print('len:',len(inputs["input_ids"][0]))
    #print('inputs tocken:',inputs)
    generated_ids = model.generate(**inputs,max_new_tokens=4096,pad_token_id=tokenizer.eos_token_id)
    #print('generated_ids:',generated_ids)
    #print(generated_ids.shape)
    
    outgen=generated_ids[:,input_ids_len:]
    print('output tocken:',outgen)
    outputs = tokenizer.batch_decode(outgen, skip_special_tokens=True)
    #print(type(outputs))
    #print(outputs)
        
    return outputs


problems = read_problems()


#problems['HumanEval/0']['prompt']='def return1():\n'
#problems['HumanEval/0']['canonical_solution']= '    return 1'

num_samples_per_task = 1
samples = []
for task_id in problems:
    print('task_id:',task_id)
    if task_id=='HumanEval/3':
        break
   
    for _ in range(num_samples_per_task):
        #print('query:',problems[task_id]["prompt"])
        ans=generate_one_completion(problems[task_id]["prompt"])
        print('Answer:',ans)
        
        samples.append(dict(task_id=task_id, completion=ans[0]))

#completion=generate_one_completion(problems[task_id]["prompt"]))
#     for task_id in problems
#     for _ in range(num_samples_per_task
#samples = [dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
#     for task_id in problems
#     for _ in range(num_samples_per_task)
# ]
print(samples)
write_jsonl("samples.jsonl", samples)
#print(problems)


# problems = read_problems()
# prompt = "Hello, my llama is cute"
# inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
# generated_ids = model.generate(**inputs)
# outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)




'''
num_samples_per_task = 200
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in problems
    for _ in range(num_samples_per_task)
]
write_jsonl("samples.jsonl", samples)







'''


'''

task_id: umanEval/0	
prompt: from typing import List def has_close_elements(numbers: List[float], threshold: float) -> bool: """ Check if in given list of numbers, are any two numbers closer to each other than given threshold. >>> has_close_elements([1.0, 2.0, 3.0], 0.5) False >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) True """	
canonical_solution: for idx, elem in enumerate(numbers): for idx2, elem2 in enumerate(numbers): if idx != idx2: distance = abs(elem - elem2) if distance < threshold: return True return False	
test: METADATA = { 'author': 'jt', 'dataset': 'test' } def check(candidate): assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False	
hentry_point: as_close_elements

DatasetDict({
    test: Dataset({
        features: ['task_id', 'prompt', 'canonical_solution', 'test', 'entry_point'],
        num_rows: 164
    })
})
'''
'''
#llama-7b   0.105   https://huggingface.co/meta-llama/Llama-2-7b-hf
#llama-13b  0.158   https://huggingface.co/meta-llama/Llama-2-13b-hf


#Code LLaMA     :https://huggingface.co/codellama/CodeLlama-7b-hf
#                https://huggingface.co/codellama/CodeLlama-7b-Python-hf


최신 bitsandbytes 라이브러리 pip install bitsandbytes>=0.39.0
최신 accelerate를 소스에서 설치 pip install git+https://github.com/huggingface/accelerate.git
최신 transformers를 소스에서 설치 pip install git+https://github.com/huggingface/transformers.git



from transformers import AutoModelForCausalLM

model_name = "bigscience/bloom-2b5"
model_4bit = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)


'''