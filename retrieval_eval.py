import json

import numpy as np

import faiss

from sentence_transformers import SentenceTransformer



faiss_index = faiss.read_index('faiss_index.index')



with open('metadata.json', 'r') as f:

    metadata = json.load(f)



embedding_model = SentenceTransformer('intfloat/e5-large-v2')



def search_faiss(query, k=10):

    query_embedding = embedding_model.encode([query.lower().strip()])

    D, I = faiss_index.search(np.array(query_embedding).astype("float32"), k=k)

    retrieved_texts = [metadata[i] for i in I[0] if i != -1]

    return retrieved_texts[:2]



def evaluate_retrieval(test_set_path, k=10):

    with open(test_set_path, 'r') as f:

        file_data = json.load(f)

        test_set = file_data['test_set']



    total_precision, total_recall, total_f1 = 0.0, 0.0, 0.0

    total_questions = len(test_set)

    results = []



    for i in test_set:

        query = i['query']

        ground_truth = set(i['related_contexts'])

        retrieved_contexts = set(search_faiss(query, k=k))

        matched_contexts = set()



        for true_context in ground_truth:

            for retrieved_context in retrieved_contexts:

                if true_context in retrieved_context or retrieved_context in true_context:

                    matched_contexts.add(true_context)



        precision = len(matched_contexts) / len(retrieved_contexts) if retrieved_contexts else 0

        recall = len(matched_contexts) / len(ground_truth) if ground_truth else 0

        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0



        total_precision += precision

        total_recall += recall

        total_f1 += f1



        results.append({

            "Query": query,

            "Ground Truth": ground_truth,

            "Retrieved Contexts": retrieved_contexts,

            "Precision": round(precision, 2),

            "Recall": round(recall, 2),

            "F1": round(f1, 2)

        })



        



    avg_precision = total_precision / total_questions if total_questions > 0 else 0

    avg_recall = total_recall / total_questions if total_questions > 0 else 0

    avg_f1 = total_f1 / total_questions if total_questions > 0 else 0




    print(f"Average Precision: {round(avg_precision, 2)}")

    print(f"Average Recall: {round(avg_recall, 2)}")

    print(f"Average F1-Score: {round(avg_f1, 2)}")



    return results, {"Precision": avg_precision, "Recall": avg_recall, "F1-Score": avg_f1}



if __name__ == "__main__":

    test_set_path = "./retrieval_test_set.json"

    results, metrics = evaluate_retrieval(test_set_path, k=10)


