import torch
import numpy as np
from scipy.stats import hmean

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def standarize_scaling(vector):
    # vector=(vector-torch.mean(vector))/torch.std(vector)
    vector=(vector-min(vector))/(max(vector)-min(vector))
    return vector

def thresholding(thresholding_mask,seen_mask,name,offset_val=0.2):
    # pair_scores= torch.load(
    #     f'utils/feasibility_pair_{name}.pt',
    #     map_location='cpu')['feasibility']

    # feasibility_score1 = torch.load(
    #     f'utils/feasibility_conceptnetRelation_{name}.pt',
    #     map_location='cpu')['feasibility']
    feasibility_score2 = torch.load(
        f'utils/feasibility_glove_{name}.pt',
        map_location='cpu')['feasibility']
    feasibility_score3 = torch.load(
        f'utils/feasibility_numbatch_{name}.pt',
        map_location='cpu')['feasibility']
    
    feasibility_score= torch.fmax(standarize_scaling(feasibility_score2),standarize_scaling(feasibility_score3))
    
    feasibe_score_for_seen=torch.masked_select(feasibility_score, thresholding_mask.bool())#feasibility_score[seen_mask]
    
    threshold=torch.min(feasibe_score_for_seen[feasibe_score_for_seen>0.])+offset_val# min(feasibe_score_for_seen)+0.3
    
    #very manual way to filter everything.
    # feasibility_score_mask=torch.clone(feasibility_score)
    # for i in range(len(feasibility_score)):
    #     if (feasibility_score[i]!=torch.tensor(0.) and feasibility_score[i]<threshold):
    #         feasibility_score_mask[i]=int(0)
    #     elif (feasibility_score[i]!=torch.tensor(0.) and feasibility_score[i]>threshold):
    #         feasibility_score_mask[i]=int(1)
    #     elif feasibility_score[i]==torch.tensor(0.):
    #         feasibility_score_mask[i]=int(1)

    #like they did it in the soft prompting
    feasibility_score_mask = (feasibility_score >= threshold).float()
    #check for training pairs.
    for i,_ in enumerate(feasibility_score_mask):
        if seen_mask[i]==1:
            feasibility_score_mask[i]=1
    print('Number of filtered combinations:',feasibility_score_mask.sum())
    return threshold,feasibility_score_mask#+thresholding_mask

def threshold_with_feasibility(
    logits,
    seen_mask,
    threshold=None,
    feasiblity=None):
    """Function to remove infeasible compositions.

    Args:
        logits (torch.Tensor): the cosine similarities between
            the images and the attribute-object pairs.
        seen_mask (torch.tensor): the seen mask with binary
        threshold (float, optional): the threshold value.
            Defaults to None.
        feasiblity (torch.Tensor, optional): the feasibility.
            Defaults to None.

    Returns:
        torch.Tensor: the logits after filtering out the
            infeasible compositions.
    """
    score = copy.deepcopy(logits)
    # Note: Pairs are already aligned here
    mask = (feasiblity >= threshold).float()
    # score = score*mask + (1.-mask)*(-1.)
    score = score * (mask + seen_mask)

    return score
