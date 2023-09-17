import numpy as np
import cohere
co = cohere.Client('hMrOlmJLtKhsrSU1kOGkE2J0ZGxoNHGbcnhYShts')

def add_embeddings(prompt, index):
    # Load dataaset
    # Add embeddings to dataset
    
    embeds = co.embed(
    texts= [prompt],
    model='embed-english-v2.0',
    # truncate='LEFT'
    ).embeddings


    shape = np.array(embeds).shape

    batch_size = 1024

    ids = [str(i) for i in range(shape[0])]

    # create list of metadata dictionaries
    meta = [{'text': text} for text in [prompt]]

    # create list of (id, vector, metadata) tuples to be upserted
    to_upsert = list(zip(ids, embeds, meta))
    
    for i in range(0, shape[0], batch_size):
        i_end = min(i+batch_size, shape[0])
        index.upsert(vectors=to_upsert[i:i_end])



def retrieve_embeddings(prompt, index):
    
    xq = co.embed(
    texts=[prompt],
    ).embeddings

    res = index.query(xq, top_k=10, include_metadata=True)

    metadata_list = [match['metadata']['text'] for match in res['matches']]
    result = co.rerank(model="rerank-english-v2.0", query=prompt, documents=metadata_list, top_n=5)
    
    # return result
    for match in res['matches']:
        print(f"{match['score']:.2f}: {match['metadata']['text']}")

    return result


def update_template(data: str):
    template = """I am visually impaired, I am trying to walk with the help of my computer vision model. The computer vision model gives me all persons and objects with their position in my view. 
    Please guide me through to my destination. All objects are infront of me so I will be walking forward.  First you will list all the objects and their position, make sure you advise me on being careful if two objects are in the same are (left, right, center). Then I will ask a question, please help me so I can find my way. If there is not question, then respond accordingly.
    {vision_info}

    {question}
    Answer:
    """