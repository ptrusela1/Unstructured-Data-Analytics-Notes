from openai import OpenAI
my_key = ''
client = OpenAI(api_key = my_key)

audio_file= open("/Users/sberry5/Documents/audio_test.mp3", "rb")

transcript = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file,
  response_format="text"
)

transcript

completion = client.chat.completions.create(
  model="gpt-4",
  messages=[
    {"role": "system", "content": "Make a rap song with the text."},
    {"role": "user", "content": "{transcript}".format(transcript = transcript)},
  ], 
  temperature=1, 
  max_tokens=150
)

print(completion.choices[0].message.content)


