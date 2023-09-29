import os
import chainlit as cl
import openai

openai.api_key = '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'

@cl.on_message
async def main(request:str):
    chatOutput = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k",
                                            messages=[{"role": "assistant", "content": "You are a helpful assistant for Data Science."},
                                                      {"role": "user", "content": request}
                                                      ]
                                            )
    await cl.Message(content = f'{chatOutput.choices[0].message.content}').send()

