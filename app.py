from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
from langchain.docstore.document import Document
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma,FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import WikipediaLoader, WebBaseLoader
from langchain.prompts.chat import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema import HumanMessage, AIMessage
import os
import re
from langsmith import Client
import time
import threading
from langchain_huggingface import HuggingFaceEmbeddings


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
#os.environ["LANGCHAIN_API_KEY"] = userdata.get("lsmith")

os.environ["LANGCHAIN_ENDPOINT"]= "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]="default"

# Initialize Groq
os.environ["GROQ_API_KEY"] = "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"
llm = ChatGroq(model="llama3-8b-8192", temperature=0.7, verbose=True, timeout=None)

# Initialize Langsmith
client = Client()
dataset_name = "FYP3 Draft Dataset"

def get_video_id(url):
    """Extract video ID from YouTube URL"""
    if 'youtu.be' in url:
        return url.split('/')[-1]
    elif 'youtube.com' in url:
        return url.split('v=')[1].split('&')[0]
    return None

def get_transcript(youtube_url, chunk_duration=10):
    try:
        # Extract Video ID from URL
        video_id = youtube_url.split("v=")[-1].split("&")[0]

        # Fetch English transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])

        # Convert to DataFrame
        df = pd.DataFrame(transcript)

        # Grouping captions into chunks of 'chunk_duration' seconds
        grouped_transcript = []
        current_chunk = {"start": 0, "text": ""}

        for _, row in df.iterrows():
            if row["start"] - current_chunk["start"] < chunk_duration:
                current_chunk["text"] += " " + row["text"]
            else:
                current_chunk["duration"] = chunk_duration
                grouped_transcript.append(current_chunk)
                current_chunk = {"start": row["start"], "text": row["text"]}

        # Add last chunk
        if current_chunk:
            current_chunk["duration"] = chunk_duration
            grouped_transcript.append(current_chunk)

        # Create new DataFrame
        new_df = pd.DataFrame(grouped_transcript)

        # Save as CSV
        new_df.to_csv("youtube_transcript_grouped.csv", index=False, encoding="utf-8")
        print("✅ Transcript saved as youtube_transcript_grouped.csv")

        return new_df
    except Exception as e:
        print(f"Error getting transcript: {e}")
        return None

def prepare_documents(df):
    texts = df['text']
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    split_docs = []
    
    for text in texts:
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            split_docs.append({"text": chunk})
    print("splitdocs",split_docs)
    return [Document(page_content=doc["text"]) for doc in split_docs]

def create_vector_store(docs):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    print("Embedding",FAISS.from_documents(docs, embedding=embedding))
    return FAISS.from_documents(docs, embedding=embedding)

def generate_qa_pairs(docs):
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    qa_pairs = []
    
    for doc in docs[:3]:  # Limit to 25 chunks
        try:
            output = qa_chain.run(input_documents=[doc], 
                                question="Generate 1 question-answer pair based on the context. The question and answer should be thought-provoking in terms of the context but the answer shouldnt be very long. Your output should be like this 'Question:xxxx,Answer:xxxx'.")
            qa_pairs.append(output)
        except Exception as e:
            print(f"Error generating QA pair: {e}")
    print("qa_pairs",qa_pairs)
    return qa_pairs

def process_qa_pairs(qa_pairs):
    inputs = []
    outputs = []
    
    for qa in qa_pairs:
        try:
            match = re.search(r"Question:\s*(.*?)\s*Answer:\s*(.*)", qa, re.DOTALL)
            if match:
                question = match.group(1).strip()
                answer = match.group(2).strip()
                inputs.append({"question": question})
                outputs.append({"answer": answer})
        except Exception as e:
            print(f"Error processing QA pair: {e}")
    
    return inputs, outputs

def save_to_langsmith(inputs, outputs):
    datasets = {d.name: d for d in client.list_datasets()}
    
    if dataset_name not in datasets:
        client.create_dataset(dataset_name)
    
    dataset_id = str(datasets[dataset_name].id)
    
    for input_data, output_data in zip(inputs, outputs):
        client.create_example(
            dataset_id=dataset_id,
            inputs={"question": input_data["question"]},
            outputs={"answer": output_data["answer"]}
        )

def create_chat_chain(vectorStore):
    print("start")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retriever = vectorStore.as_retriever()
    print("end")
    return create_retrieval_chain(retriever, document_chain)

def get_wikipedia_data(query):
    try:
        data = WikipediaLoader(query=query, load_max_docs=2).load()
        return data[0].metadata["source"]
    except Exception as e:
        print(f"Error getting Wikipedia data: {e}")
        return None

def update_vector_store(vectorStore, link):
    try:
        loader = WebBaseLoader(link)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
        splitDocs = text_splitter.split_documents(docs)
        vectorStore.add_documents(splitDocs)
        return vectorStore
    except Exception as e:
        print(f"Error updating vector store: {e}")
        return vectorStore

from flask import copy_current_request_context

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        youtube_url = request.form.get('youtube_url')
        video_id = get_video_id(youtube_url)
        
        if not video_id:
            return render_template('index.html', error="Invalid YouTube URL")

        session.clear()
        session['video_id'] = video_id
        session['youtube_url'] = youtube_url
        session.modified = True

        @copy_current_request_context
        def run_process_video():
            process_video(youtube_url)

        # ✅ Start thread with wrapped function
        try:
            thread = threading.Thread(target=run_process_video)
            thread.start()
        except Exception as e:
            session['error'] = f"Failed to start processing: {str(e)}"
            session.modified = True

        return render_template('index.html', processing=True, youtube_url=youtube_url)

    return render_template('index.html')
    


from flask import copy_current_request_context

def process_video(url):
    try:
        df = get_transcript(url)
        if df is None:
            session['error'] = "Failed to get transcript"
            session.modified = True
            return
        
        docs = prepare_documents(df)
        vectorStore = create_vector_store(docs)
        qa_pairs = generate_qa_pairs(docs)
        inputs, outputs = process_qa_pairs(qa_pairs)
        save_to_langsmith(inputs, outputs)
        chain = create_chat_chain(vectorStore)

        session['vectorStore'] = vectorStore
        session['chain'] = chain
        session['processed'] = True
        session['qa_pairs'] = qa_pairs
        session.modified = True
        print("route end")
    except Exception as e:
        print(f"Error processing video: {e}")
        session['error'] = str(e)
        session.modified = True


@app.route('/status')
def status():
    if 'processed' in session:
        return jsonify({'status': 'complete', 'qa_pairs': session.get('qa_pairs', [])})
    elif 'error' in session:
        return jsonify({'status': 'error', 'message': session['error']})
    else:
        return jsonify({'status': 'processing'})

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'chain' not in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        user_input = request.form.get('message')
        wikipedia_query = request.form.get('wikipedia_query')
        
        chat_history = session.get('chat_history', [])
        chain = session['chain']
        vectorStore = session['vectorStore']
        
        if wikipedia_query:
            link = get_wikipedia_data(wikipedia_query)
            if link:
                vectorStore = update_vector_store(vectorStore, link)
                session['vectorStore'] = vectorStore
                session['chain'] = create_chat_chain(vectorStore)
                chain = session['chain']
                chat_history.append(HumanMessage(content=f"Wikipedia search: {wikipedia_query}"))
                chat_history.append(AIMessage(content=f"I've updated my knowledge with information from Wikipedia about {wikipedia_query}"))
        
        response = chain.invoke({
            "chat_history": chat_history,
            "input": user_input,
        })
        
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response["answer"]))
        session['chat_history'] = chat_history
        
        return jsonify({
            'response': response["answer"],
            'chat_history': [{'role': 'human' if isinstance(msg, HumanMessage) else 'ai', 'content': msg.content} for msg in chat_history]
        })
    
    return render_template('chat.html', youtube_url=session.get('youtube_url'))

# Audio transcription routes
@app.route('/audio_transcribe', methods=['POST'])
def audio_transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded."}), 400
    
    audio_file = request.files['audio']
    
    try:
        # Here you would integrate your FinetuneWhisper code
        # For now, we'll just simulate a response
        transcription = "This is a simulated transcription of the audio."
        return jsonify({"transcription": transcription})
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return jsonify({"error": "An error occurred while processing the audio."}), 500

if __name__ == '__main__':
    app.run(debug=True)
