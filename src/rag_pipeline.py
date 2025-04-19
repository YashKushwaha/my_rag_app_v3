def generate_answer(question, embedder, vectorstore, llm):
    query_embedding = embedder.embed(question)
    relevant_chunks = vectorstore.retrieve(query_embedding)
    context = "\n".join(relevant_chunks)
    prompt = f"Answer the question using the context below:\nContext:\n{context}\n\nQuestion: {question}"
    response = llm.generate(prompt)
    return response

