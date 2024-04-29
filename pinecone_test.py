from pinecone import Pinecone, ServerlessSpec
from pinecone_datasets import load_dataset
import pandas as pd
import numpy as np
import time

BATCH_SIZE = 100
PINECONE_API_KEY = "284aedd1-2870-4523-8656-9110293fce67"

def upload_data(dataset, index):
    print(f"\n Uploading Data...")
    start = time.perf_counter()
    index.upsert_from_dataframe(dataset.drop(columns=["sparse_values","metadata","blob"]))
    end = time.perf_counter()
    return (end-start)/60.0



def query(query_vectors, index, k=100):
    print(f"\n Batch Querying...")
    times = []
    results = []
    for i in range(0,10000):
        item = query_vectors[i]
        start = time.perf_counter()
        res = index.query(vector=item, top_k=k, include_values=True)
        end = time.perf_counter()
        results.append(res)
        times.append(end-start)
        if i%1000==0: print("batch done", i)
    return [times, results]

def formatResults(results):
	formatted_results = []
	for result in results:
		matches = result["matches"]
		res_matches = []
		for match in matches:
			id = match["id"]
			score = match["score"]
			values = match["values"]
			x = {"id":id, "score":score, "values":values}
			res_matches.append(x)
		formatted_results.append(res_matches)
    
	return formatted_results


def main():
	glove100_dataset = load_dataset("ANN_GloVe_d100_angular")
	print("starting pinecone")
	pc = Pinecone(api_key=PINECONE_API_KEY)
	pc.delete_index("glove100d-aws")
	pc.create_index(
		name="glove100d-aws",
		dimension=100,
		metric="cosine",
		# change later, serverless only available on AWS 
		# trying to start with some boilerplate code rn git commit -am ""
		spec=ServerlessSpec(
			cloud='aws', 
			region='us-east-1'
		) 
	) 

	index = pc.Index("glove100d-aws")
	print("created index")
	print("making datatset")
	dataset = glove100_dataset.documents
	print("made dataset")
	dl = len(dataset)
	partition_len = int(dl*0.9)
	df_1 = dataset.iloc[:partition_len,:]
	df_2 = dataset.iloc[partition_len:,:]
	upload_latency = upload_data(df_1, index)
	upload_latency = upload_data(df_2, index)
	query_vectors = [item.tolist() for item in glove100_dataset.queries["vector"]]
	times, results = query(query_vectors, index, 100)
	print("Mean query latency",np.mean(times))
	nn = glove100_dataset.queries["blob"][0]["nearest_neighbors"]
	formatted_results = formatResults(results)
	nn100 = []
	for x in glove100_dataset.queries["blob"]: 
		s = set(x["nearest_neighbors"])
		nn100.append(s)
	recall_per_query_100 = []
	for idx,qr in enumerate(formatted_results):
	    true_pos = 0
	    for res in qr:
	        if res["id"] in nn100[idx]: true_pos+=1
	    recall_per_query_100.append(true_pos/100)
	print("Avg recall: ", np.mean(recall_per_query_100))

if __name__ == "__main__":
    main()
