import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Iterable
from pycocoevalcap.cider.cider import Cider
#from modules.tokenization import BertTokenizer
from CLIP4Caption.modules.tokenization import BertTokenizer

def calculate_cider(gt_dict: Dict[int, List], prediction_dict: Dict[int, List]) -> Tuple[float, Iterable]:
    """Calculate CIDer score
    Args:
        gt_dict (Dict[int, List]): each key contains a list of captions
        prediction_dict (Dict[int, List]): each key contain a list of a SINGLE caption
    Returns:
        Tuple[float,Iterable]:
            - avg score of all keys
            - score for each key
    """
    scorer = Cider()
    return scorer.compute_score(gt_dict, prediction_dict)


def idx_to_token(idx: Iterable[int], tokenizer: BertTokenizer) -> str:
    """Convert indexes into word
    Args:
        idx (Iterable[int]):
        tokenizer (BertTokenizer)
    Returns:
        str:
    """
    decode_text_list = tokenizer.convert_ids_to_tokens(idx)
    total_len = len(idx)
    if '[SEP]' in decode_text_list:
        total_len = decode_text_list.index('[SEP]')
        decode_text_list = decode_text_list[:total_len]
    if '[PAD]' in decode_text_list:
        total_len = decode_text_list.index('[PAD]')
        decode_text_list = decode_text_list[:total_len]
    mask = [1] * total_len + [0] * (len(idx) - total_len)
    decode_text = ' '.join(decode_text_list)
    decode_text = decode_text.replace(' ##', '').strip('##').strip()
    return decode_text, mask


def sample_caption(log_prob: torch.Tensor, multinomial: bool = False) -> torch.Tensor:
    """Sample generated caption based on probability distribution
    Args:
        log_prob (torch.Tensor): (BATCH_SIZE, MAX_WORDS, VOCAB_SIZE)
        multinomial (bool): whether to sample using multinomial or argmax (greedy)
    Returns:
        torch.Tensor: (BATCH_SIZE, MAX_WORDS)
    """
    if multinomial:
        batch_size, max_words, _ = log_prob.shape
        result = torch.empty(batch_size, max_words, dtype=torch.int64).cuda()
        for i in range(max_words):
            result[:, i] = torch.multinomial(log_prob[:, i, :], 1).squeeze()
    else:
        result = torch.argmax(log_prob, dim=2)
    
    return result

def calculate_reward(
    decoder_scores: torch.Tensor,
    pairs_output_caption_ids: torch.Tensor,
    tokenizer: BertTokenizer,
    multiplier: float = 1.0,
    retrieval: bool = False,
) -> torch.Tensor:
    """Calculate rewards based on cider score of prediction relative to gt
    Args:
        decoder_scores (torch.Tensor): (BATCH_SIZE, MAX_WORDS, VOCAB_SIZE)
        pairs_output_caption_ids (torch.Tensor): (BATCH_SIZE, MAX_WORDS)
        tokenizer (BertTokenizer)
        multiplier (float): to scale the reward, defaults to 1
    Returns:
        torch.Tensor: (1)
    """
    batch_size = decoder_scores.shape[0]
    prob = F.softmax(decoder_scores, dim=2)
    argmax_seq = sample_caption(prob, False)
    multinomial_seq = sample_caption(prob, True) # (BATCH_SIZE, MAX_WORDS)
    log_prob = torch.gather(torch.log(prob), 2, multinomial_seq.unsqueeze(-1)).squeeze()  # (BATCH_SIZE, MAX_WORDS)

    gt_dict = {}
    pred_dict_argmax = {}
    pred_dict_multinomial = {}
    reward_mask = []
    for i in range(batch_size):
        pred_tokens_argmax = argmax_seq[i].cpu().detach().numpy()
        pred_tokens_multinomial = multinomial_seq[i].cpu().detach().numpy()
        gt_tokens = pairs_output_caption_ids[i].cpu().detach().numpy()

        temp_word, _ = idx_to_token(gt_tokens, tokenizer)
        gt_dict[i] = [temp_word]

        temp_word, _ = idx_to_token(pred_tokens_argmax, tokenizer)
        pred_dict_argmax[i] = [temp_word]

        temp_word, mask = idx_to_token(pred_tokens_multinomial, tokenizer)
        pred_dict_multinomial[i] = [temp_word]

        reward_mask.append(mask)

    _, cider_scores = calculate_cider(gt_dict, pred_dict_multinomial)
    _, cider_scores_baseline = calculate_cider(gt_dict, pred_dict_argmax)
    reward_mask = torch.FloatTensor(reward_mask).cuda()  # (BATCH_SIZE, MAX_WORDS)
    cider_scores = torch.FloatTensor(cider_scores).cuda()
    cider_scores_baseline = torch.FloatTensor(cider_scores_baseline).cuda()

    rewards = torch.einsum('nm,n->nm', [log_prob, cider_scores - cider_scores_baseline])
    rewards *= reward_mask
    rewards *= multiplier
    rewards = torch.sum(rewards) / torch.sum(reward_mask)
    if retrieval:
        return cider_scores,cider_scores_baseline,pred_dict_multinomial,pred_dict_argmax,reward_mask,log_prob
    else:
        return rewards


if __name__ == '__main__':
    gt_dict = {
        1: ['a man is eating a salad', 'a guy is consuming food', 'a person is eating spinach'],
        2: ['a girl is swimming', 'a girl is in the water', 'a person is swimming in the swimming pool']
    }
    prediction_dict = {
        1: ['a boy is eating'],
        2: ['a woman is swimming'],
    }
    base_line_dict = gt_dict.copy()

    cider_score = calculate_cider(gt_dict, prediction_dict)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    decoder_scores = torch.rand(3, 4, 30000).cuda()
    pairs_output_caption_ids = torch.randint(0, 30000, (3, 4)).cuda()
    reward = calculate_reward(decoder_scores, pairs_output_caption_ids, tokenizer)
    
