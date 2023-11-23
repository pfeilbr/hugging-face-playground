from transformers import pipeline, set_seed, AutoModelForCausalLM, AutoTokenizer

def test_text_generation():
    generator = pipeline('text-generation', model='gpt2')
    set_seed(42)
    output = generator("Hello, I like to play cricket,", max_length=60, num_return_sequences=7)
    print(output)


def test_automatic_speech_recognition():
    transcriber = pipeline(task="automatic-speech-recognition")
    #resp = transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
    resp = transcriber("https://audio-samples.github.io/samples/mp3/blizzard_unconditional/sample-0.mp3")
    print (resp)

# https://huggingface.co/docs/transformers/pipeline_tutorial
def test_openai_whisper_speech_to_text():
    transcriber = pipeline(model="openai/whisper-large-v2", device_map="auto")
    resp = transcriber("https://audio-samples.github.io/samples/mp3/blizzard_unconditional/sample-0.mp3")
    print (resp)

# https://huggingface.co/docs/transformers/llm_tutorial
def test_local_llm_mistralai():
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto", load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")
    model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to("cuda")
    generated_ids = model.generate(**model_inputs)
    resp = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print (resp)

if __name__ == '__main__':
    test_text_generation()
    #test_automatic_speech_recognition()
    #test_openai_whisper_speech_to_text()
    #test_local_llm_mistralai()
