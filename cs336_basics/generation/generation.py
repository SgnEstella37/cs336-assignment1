import torch
from cs336_basics.model.transformer import TransformerLM
from cs336_basics.tokenization.tokenizer import Tokenizer
from cs336_basics.model.layers import softmax

def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float=1.0,
    top_p: float=1.0,
) -> str:
    device = next(model.parameters()).device
    
    token_ids=tokenizer.encode(prompt)
    input_ids=torch.tensor([token_ids],dtype=torch.long,device=device)
    
    model.eval()
    
    generated_count=0
    eos_token_ids = tokenizer.encode("<|endoftext|>")
    assert len(eos_token_ids) == 1
    eos_token_id = eos_token_ids[0]
    
    with torch.no_grad():
        while generated_count < max_new_tokens:
            model_input = input_ids[:, -model.context_length:]
            logits = model(model_input)
            next_logis=logits[:,-1,:]
            
            #temperature scaling
            if temperature <= 0:
                raise ValueError("temperature must be positive")
            scaled_logits=next_logis/temperature
            probs=softmax(scaled_logits,dim=-1)
            
            #necleus
            if not (0 < top_p <= 1):
                raise ValueError("top_p must be in (0, 1]")
            sorted_probs, sorted_indices= torch.sort(
                probs,dim=-1,descending=True
            )

            cumulative_probs=torch.cumsum(sorted_probs,dim=-1)
            mask = cumulative_probs >= top_p
            first_idx = torch.argmax(mask.int(), dim=-1)
            keep_mask = (
                torch.arange(
                    sorted_probs.shape[-1],
                    device=sorted_probs.device
                )
                <= first_idx.unsqueeze(-1)
            )
            filtered_probs = sorted_probs.masked_fill(~keep_mask, 0.0)
            filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
            sampled_idx = torch.multinomial(filtered_probs, num_samples=1)
            next_token = sorted_indices.gather(dim=-1, index=sampled_idx)
            
            if next_token.item()==eos_token_id:
                break
            
            input_ids=torch.cat([input_ids,next_token],dim=1)
            
            generated_count+=1
            
    return tokenizer.decode(input_ids[0, len(token_ids):].tolist())        
            
            
            
    