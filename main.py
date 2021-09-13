from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
output = generator("Hello, I like to play cricket,", max_length=60, num_return_sequences=7)
print(output)
