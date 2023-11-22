from transformers import pipeline, set_seed

def test_text_generation():
    generator = pipeline('text-generation', model='gpt2')
    set_seed(42)
    output = generator("Hello, I like to play cricket,", max_length=60, num_return_sequences=7)
    print(output)


def test_automatic_speech_recognition():
    transcriber = pipeline(task="automatic-speech-recognition")
    resp = transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
    print (resp)

if __name__ == '__main__':
    test_text_generation()
    test_automatic_speech_recognition()
