from dotenv import find_dotenv,load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI
import streamlit as st
import os
import requests

# Load environment variables from a .env file
load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Function to convert an image to a story
def img2story(url):
    # Create a pipeline for image-to-text conversion
    pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = pipe(url)[0]['generated_text']
    print(text)
    return text

# Function to generate a story based on a scenario
def gen_story(scenario):
    # Define a template for story generation
    template = """ You are a story teller.
    You can generate a simple story not more than about 50 words.
    Context = {scenario}
    STORY:

    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    # Create a language model chain for story generation
    llm = LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=1), prompt=prompt, verbose=True)

    story = llm.predict(scenario=scenario)
    print(story)
    return story

# Function to convert text to speech
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payloads = {
        "inputs": message
    }
    response = requests.post(API_URL, headers=headers, json=payloads)
    with open("audio.flac", "wb") as file:
        file.write(response.content)

# Main function to create the Streamlit app
def main():
    st.set_page_config(page_title="Image to Story", page_icon="")
    st.header("Turn img into an audio story")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        scenario = img2story(uploaded_file.name)
        story = gen_story(scenario)
        text2speech(story)
        
        with st.expander("Scenario"):
            st.write(scenario)
        with st.expander("Story"):
            st.write(story)
        st.audio("audio.flac")

if __name__ == "__main__":
     main()
