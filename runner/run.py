from vllm import LLM, SamplingParams
llm = LLM("Qwen/Qwen2.5-3B-Instruct", max_model_len=4096, gpu_memory_utilization=0.85, seed=0)

def run(h,item_ids,temperature,n_samples):
    sp = SamplingParams(temperature=temperature,n=n_samples,max_tokens=512,seed = 0)
    convs =[h.message(ds[i]["question"]) for i in item_ids]
    outs = llm.chat(convs, sampling_params=sp)
    rows = []
    for i,o in zip(item_ids,outs):
        for s,c in enumerate(o.outputs):
            rows.append(dict(harness = h,id=item_ids[i], temp = temperature, sample = s))
    return rows 
    