import json

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from rouge import Rouge

from llm import generate_response, search_faiss



def rouge_bleu_eval(test_set_path):

    with open(test_set_path, "r") as f:

        test_set = json.load(f)



    total_bleu = 0

    total_rouge_1 = 0

    total_rouge_2 = 0

    total_rouge_l = 0



    rouge = Rouge()

    count = len(test_set)



    for test in test_set:

        query = test["query"]

        expected_answer = test["expected_answer"]



        retrieved_contexts = search_faiss(query)

        generated_response = generate_response(query, retrieved_contexts)



        bleu_score = sentence_bleu(

            [expected_answer.split()],

            generated_response.split(),

            smoothing_function=SmoothingFunction().method1

        )



        rouge_scores = rouge.get_scores(generated_response, expected_answer, avg=True)



        total_bleu += bleu_score

        total_rouge_1 += rouge_scores["rouge-1"]["f"]

        total_rouge_2 += rouge_scores["rouge-2"]["f"]

        total_rouge_l += rouge_scores["rouge-l"]["f"]



    avg_bleu = total_bleu / count

    avg_rouge_1 = total_rouge_1 / count

    avg_rouge_2 = total_rouge_2 / count

    avg_rouge_l = total_rouge_l / count



    print(f"\nAverage BLEU Score: {avg_bleu:.2f}")

    print(f"Average ROUGE-1 F-Measure: {avg_rouge_1:.2f}")

    print(f"Average ROUGE-2 F-Measure: {avg_rouge_2:.2f}")

    print(f"Average ROUGE-L F-Measure: {avg_rouge_l:.2f}")





if __name__ == "__main__":

    test_set_path = "text_eval_set.json"

    rouge_bleu_eval(test_set_path)


