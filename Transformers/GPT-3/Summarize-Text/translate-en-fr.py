import os
import openai

openai.api_key = "YOUR API KEY"

text = "While all prompts result in completions, it can be helpful to think of text completion as its own task in instances where you want the API to pick up where you left off. "

response = openai.Completion.create(

    engine='davinci',
    prompt="I do not speak French.\nFrench: Je ne parle pas français.\n\nEnglish: See you later!\nFrench: À tout à l'heure!\n\nEnglish: Where is a good restaurant?\nFrench: Où est un bon restaurant?\n\nEnglish: What rooms do you have available?\nFrench: Quelles chambres avez-vous de disponible?\n\nEnglish: Where is the restroom?\nFrench:",
    temperature=0.5,
    max_tokens=250,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["\n"]
)

print(response["choices"][0]["text"])