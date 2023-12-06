from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import json
import time
from openai import OpenAI


KEY = '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'

client = OpenAI(api_key = KEY)

def get_completion(prompt, model='gpt-3.5-turbo-1106'):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model=model,messages=messages)
    return response.choices[0].message.content

if __name__ == "__main__":

    # Get the Video2Commonsense output from json files
    with open('att_test_results.json') as f:
        att_data = json.load(f)

    with open('eff_test_results.json') as f:
        eff_data = json.load(f)

    with open('int_test_results.json') as f:
        int_data = json.load(f)

    # Create a prompt for the pre-trained generative transformer
    prompt_start = "Create a single sentence video caption explaining what happened in the video based on the following information."
    # Dictionary containing the ground truth
    ground_truth = {}
    # Dictionary of generated captions
    generated_caption = {}
    count = 1
    for video_key in att_data['data'].keys():
        if count < 553:
            count += 1
            continue
        if count % 3 == 0:
            print("\nWaiting for 60 seconds to not exceed rate limits")
            time.sleep(60)
        count += 1

        prompt = prompt_start

        # attribute data
        att_data_from_video = att_data['data'][video_key]
        prompt += "\nDescription of the person: " + att_data_from_video[0]['pred_caption']

        # Effect data
        eff_data_from_video = eff_data['data'][video_key]
        prompt += "\nWhat changes due to the action: " + eff_data_from_video[0]['pred_caption']
        # Intention data
        int_data_from_video = int_data['data'][video_key]
        prompt += "\nWhy the action is taking place: " + int_data_from_video[0]['pred_caption']

        # Get the response from the pre-trained generative transformer
        response = get_completion(prompt)

        print("\n-----", video_key, "-----\nGround truth:", att_data_from_video[0]['gt_caption'])
        print("Attribute:" + att_data_from_video[0]['pred_caption'])
        print("Effect:" + eff_data_from_video[0]['pred_caption'])
        print("Intention" + int_data_from_video[0]['pred_caption'])
        print("Created caption:", response)

        # Write the video key, generated caption, and ground truth caption to a text file
        f = open("final_captions.txt", 'a')
        f.write(video_key + "_" + response + "_" + att_data_from_video[0]['gt_caption'] + '\n')
        f.close()

        # Add the result to the ground_truth and generated_caption dictionaries for evaluation
        ground_truth[video_key] = [att_data_from_video[0]['gt_caption']]
        generated_caption[video_key] = [response]

    print("\n\nRESULTS\n")
    avg_bleu_score, bleu_scores = Bleu(4).compute_score(ground_truth, generated_caption)
    avg_cider_score, cider_scores = Cider().compute_score(ground_truth, generated_caption)
    avg_rouge_score, rouge_scores = Rouge().compute_score(ground_truth, generated_caption)
    print("BLEU:", 100 * avg_bleu_score)
    print("cIDER:", 100 * avg_cider_score)
    print("rouge:", 100 * avg_rouge_score)
