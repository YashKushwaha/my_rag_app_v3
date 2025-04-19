def generate_answer(question, embedder, vectorstore, llm):
    query_embedding = embedder.embed(question)
    relevant_chunks = vectorstore.retrieve(query_embedding)
    context = "\n".join(relevant_chunks)
    prompt = f"Answer the question using the context below:\nContext:\n{context}\n\nQuestion: {question}"
    response = llm.generate(prompt)
    return response

def format_history(history):
    return "\n".join([f"{h['role'].capitalize()}: {h['content']}" for h in history])

def generate_answer_with_history(question, embedder, vectorstore, llm, chat_history=None):
    query_embedding = embedder.embed(question)
    relevant_chunks = vectorstore.retrieve(query_embedding)
    context = "\n".join(relevant_chunks)
    history_text = ""

    if chat_history:
            for message in chat_history:
                history_text += f"{message['role'].capitalize()}: {message['content']}\n"

    prompt = (
        f"You are a helpful assistant.\n"
        f"{history_text}\n\n"
        f"Answer the question using the context below:\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    response = llm.generate(prompt)
    return response
