from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import json

import numpy as np

if __name__ == "__main__":

    # Get the Video2Commonsense output from json files
    with open('att_test_results.json') as f:
        att_data = json.load(f)

    with open('eff_test_results.json') as f:
        eff_data = json.load(f)

    with open('int_test_results.json') as f:
        int_data = json.load(f)

    file = open('final_captions.txt', 'r')

    # Dictionary containing the ground truth
    ground_truth = {}
    # Dictionary of generated captions
    generated_caption = {}

    for line in file:
        items = line.split('_')
        key, pred, gt = items[0], items[1], items[2]
        att_cms_gt = att_data['data'][key][0]['gt_knowledge']
        eff_cms_gt = eff_data['data'][key][0]['gt_knowledge']
        int_cms_gt = int_data['data'][key][0]['gt_knowledge']
        full_gt = [gt] + att_cms_gt + eff_cms_gt + int_cms_gt
        ground_truth[key] = full_gt
        generated_caption[key] = [pred]

    print("RESULTS\n")
    avg_bleu_score, bleu_scores = Bleu(4).compute_score(ground_truth, generated_caption)
    avg_cider_score, cider_scores = Cider().compute_score(ground_truth, generated_caption)
    avg_rouge_score, rouge_scores = Rouge().compute_score(ground_truth, generated_caption)
    print("BLEU:", 100 * np.mean(avg_bleu_score))
    print("cIDER:", 100 * avg_cider_score)
    print("rouge:", 100 * avg_rouge_score)