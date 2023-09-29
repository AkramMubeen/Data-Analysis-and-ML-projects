import re
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models.openai import ChatOpenAI


def llm(url_list, prompt):
    openai = ChatOpenAI(
            model_name="gpt-3.5-turbo-16k",
            temperature=0.5,
            streaming=True,
            max_tokens = 16000
    )
    loader_list = []
    for i in url_list:
        loader_list.append(WebBaseLoader(i))
    index = VectorstoreIndexCreator().from_loaders(loader_list)
    ans = index.query(prompt)
    return ans

def separate(user_links):
    # Regex pattern to match links
    link_pattern = r"(https?://\S+)"
    # Find all links in the text
    links = re.findall(link_pattern, user_links)
    # Return the list of links
    return links    
    
prompt = """As an experienced content writer, your task is to create a compelling content post  that provides a summary of a given content. The blog post should clearly explain what the content is, what it does, and how to use it. Additionally, it should address any other important questions that may arise regarding the content.

Your content engaging and informative, capturing the reader's attention and providing them with valuable insights."""

def main():
    load_dotenv()
    st.set_page_config(page_title="BLOGGY",
                       page_icon="🐸",
                       layout="wide")
    
    st.header("BLOGGY 🐸 ☕️")
    st.subheader("Well, folks, this app right here takes your pasted links and turns 'em into some nifty content using all the info it snags from those links!🐸") 
    user_links = st.text_area("Paste your links here:",placeholder="Enter your links here \n1\n2\n3")
    
    with st.sidebar:
        st.subheader("Hey there, pal! Just give ol' Kermit a little clickety-click on that 'Write' button!🐸🖱️")
        para = ""
        if st.button("Write"):
            with st.spinner("Your content is getting ready, folks! 🐸✨"):
                gif_runner = st.image("gif/froggy.gif")
                
                url_list = separate(user_links)
                para = llm(url_list,prompt)
                
                gif_runner.empty()
            with st.expander('Links history'): 
                st.info(url_list)
            with st.expander('Content history'): 
                st.info(para)    
            st.success(body="Done! Your content is ready, folks!", icon="🎉")

    st.text_area(label="",value=para, height=250,placeholder="Your content")
    



if __name__ == '__main__':
    main()        