# Amplify Test Data


## amplify_output

```python
from safetensors.torch import save_file


# model ouputs for this sequence
sequence = "MSVVGIDLGFQSCYVAVARAGGIETIANEYSDRCTPACISFGPKNR"

model = AutoModel.from_pretrained( "AMPLIFY_120M")
tokenizer = AutoTokenizer.from_pretrained(AMPLIFY_120M")
model = model.to("cpu")

input = tokenizer.encode(sequence, return_tensors="pt")
input = input.to("cpu")
output = model(input, output_attentions=True,  output_states=True)


tensors_dict = {}
# For hidden states
if output.hidden_states is not None:
    if isinstance(output.hidden_states, (list, tuple)):
        for i, hidden in enumerate(output.hidden_states):
            tensors_dict[f'hidden_states_{i}'] = hidden
    else:
        tensors_dict['hidden_states'] = output.hidden_states

# For attentions
if output.attentions is not None:
    for i, attn in enumerate(output.attentions):
        tensors_dict[f'attention_{i}'] = attn

# Save to file
save_file(tensors_dict, "amplify_output.safetensors")
```
