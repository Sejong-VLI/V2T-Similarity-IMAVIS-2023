from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
from torch.utils.data import (SequentialSampler)
import numpy as np
import random
from copy import deepcopy
import os
from collections import OrderedDict
from nlgeval import NLGEval
import time
import argparse
from tqdm import tqdm

from <VIDEOCAPTIONINGFOLDER>.modules.tokenization import BertTokenizer
from <VIDEOCAPTIONINGFOLDER>.modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from <VIDEOCAPTIONINGFOLDER>.modules.modeling import CaptionGenerator
from <VIDEOCAPTIONINGFOLDER>.modules.optimization import BertAdam
from <VIDEOCAPTIONINGFOLDER>.modules.beam import Beam
from torch.utils.data import DataLoader

from <VIDEOCAPTIONINGFOLDER>.feature_extractor.util import get_logger
from tqdm import tqdm

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice import Spice

# retrieval
from <VIDEORETRIEVALFOLDER>.modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from <VIDEORETRIEVALFOLDER>.modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from <VIDEORETRIEVALFOLDER>.modules.modeling import <VIDEORETRIEVALFOLDER>
from <VIDEORETRIEVALFOLDER>.modules.optimization import BertAdam

from <VIDEORETRIEVALFOLDER>.metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim

from <VIDEOCAPTIONINGFOLDER>.dataloaders.data_dataloader import DATALOADER_DICT
from retrieval_utils import *
torch.distributed.init_process_group(backend="nccl")

from policy_gradient import calculate_reward

global logger
isUniVL = False
def get_args(description='CaptionGenerator'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--retrieval_only", action='store_true', help="Whether to run retrieval only.")
    parser.add_argument('--data_path', type=str, default='data/MSRVTT_data.json',
                        help='caption and transcription file path')
    parser.add_argument('--features_path', type=str, default='data/msrvtt_features.pickle',
                        help='feature path for CLIP features')
    parser.add_argument('--inference_path', type=str, default=None,
                        help='inference path if None use all videos')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--gamma_rl', type=float, default=0.3, help='percentage contribution of RL')
    parser.add_argument('--reward_multiplier', type=float, default=0.1, help='scale the reward')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=1969, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--min_time', type=float, default=5.0, help='Gather small clips')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--bert_model", default="bert-model", type=str, required=True, help="Bert pre-trained model")
    parser.add_argument("--visual_model", default="visual-base", type=str, required=False, help="Visual module")
    parser.add_argument("--decoder_model", default="decoder-base", type=str, required=False, help="Decoder module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset `msrvtt` to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=0.1, help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether use MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=6, help="Layer NO. of visual.")
    parser.add_argument('--decoder_num_hidden_layers', type=int, default=3, help="Layer NO. of decoder.")

    parser.add_argument('--patience', type=int, default=80, help="Number of epochs with no improvement after which training will be stopped.")
    parser.add_argument('--patience_metric', type=str, default="CIDEr", help="Metric which is used for early stopping.")
    parser.add_argument('--target_metric', type=str, default="CIDEr", help="Target metric which is used to select the best model.")
    
    parser.add_argument("--train_type", default="joint", type=str, help="retrieval/captioning/joint/freeze_retrieval/freeze_captioning")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument('--sim_header', type=str, default="meanP",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
                        help="choice a similarity header.")
    parser.add_argument('--loose_type', action='store_true', help="Default using tight type for retrieval.")
    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument("--output_dir_retrieval", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")
    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")
    parser.add_argument("--model_file", default="", type=str, required=False, help="pretrain captioning ")
    parser.add_argument("--model_file_retrieval", default="", type=str, required=False, help="pretrain retrieval ")
    parser.add_argument("--retrieval_weight", default=0.5, type=float, required=False, help="weight for retrieval")
    parser.add_argument("--eval_type", default="captioning", type=str, help="retrieval/captioning")
    args = parser.parse_args()

    if args.sim_header == "tightTransf":
        args.loose_type = False
    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.output_dir_retrieval):
        os.makedirs(args.output_dir_retrieval, exist_ok=True)

    
    logger_retrieval = get_logger(os.path.join(args.output_dir_retrieval, "log.txt"))
    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args
def init_model(args, device, n_gpu, local_rank):

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = CaptionGenerator.from_pretrained(args.bert_model, args.visual_model, args.decoder_model,
                                   cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)



    model.to(device)


    return model

def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):

    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    no_decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    no_decay_bert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." in n]
    no_decay_nobert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." not in n]

    decay_bert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." in n]
    decay_nobert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." not in n]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in no_decay_bert_param_tp], 'weight_decay': 0.01, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_nobert_param_tp], 'weight_decay': 0.01},
        {'params': [p for n, p in decay_bert_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_nobert_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_linear', t_total=num_train_optimization_steps, weight_decay=0.01,
                         max_grad_norm=1.0)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)

    return optimizer, scheduler, model




def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        #(Spice(), "SPICE")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)

        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores



def convert_state_dict_type(state_dict, ttype=torch.FloatTensor):
    if isinstance(state_dict, dict):
        cpu_dict = OrderedDict()
        for k, v in state_dict.items():
            cpu_dict[k] = convert_state_dict_type(v)
        return cpu_dict
    elif isinstance(state_dict, list):
        return [convert_state_dict_type(v) for v in state_dict]
    elif torch.is_tensor(state_dict):
        return state_dict.type(ttype)
    else:
        return state_dict

def save_model(epoch, args, model, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file

def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = CaptionGenerator.from_pretrained(args.bert_model, args.visual_model, args.decoder_model,
                                       cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)
    else:
        model = None
    return model

def train_epoch(epoch, args, model, retrieval,train_dataloader, tokenizer,cliptokenizer, device, n_gpu, optimizer, scheduler,
                global_step, nlgEvalObj=None, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    retrieval.eval()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0
    reward = 0

    for step, batch in enumerate(train_dataloader):
        # if n_gpu == 1:
        #     # multi-gpu does scattering it-self
        #     batch = tuple(t.to(device) for t in batch)
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, video, video_mask, \
        pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids= batch




        

        # pairs_masked_text, pairs_token_labels, masked_video, video_labels_index,
        # input_mask, segment_ids
        # the above data are unnecessary to train <VIDEOCAPTIONINGFOLDER>
        # input_ids (BATCHSIZE,1,MAX WORDS)
        decoder_scores = model(video, video_mask,
                     input_caption_ids=pairs_input_caption_ids, decoder_mask=pairs_decoder_mask)
        pairs_output_caption_ids = pairs_output_caption_ids.view(-1, pairs_output_caption_ids.shape[-1])
        # decoder_scores (BATCH_SIZE, MAX_WORDS, VOCAB_SIZE)
        # pairs_output_caption_ids (BATCH_SIZE, MAX_WORDS)
        loss = model.module.decoder_loss_fct(decoder_scores.view(-1, model.module.bert_config.vocab_size), pairs_output_caption_ids.view(-1))
        loss_b_retrieval = 0
        loss_s_retrieval = 0
        loss_retrieval = 0



        if epoch > 10:
            if args.retrieval_weight <= 0:
                reward= calculate_reward(decoder_scores, pairs_output_caption_ids, tokenizer, args.reward_multiplier,False)

            else:


                


                cider_scores,cider_scores_baseline,generated_caption,greedy_caption,reward_mask,log_prob= calculate_reward(decoder_scores, pairs_output_caption_ids, tokenizer, args.reward_multiplier,True)
        
                generated_input_ids, generated_input_mask, generated_segment_ids = gc_to_idx(generated_caption,cliptokenizer,args.max_words)
            
                baseline,baseline_mask, baseline_segments =  gc_to_idx(greedy_caption,cliptokenizer,args.max_words)

            

                with torch.no_grad():

                    
                        loss_gt_retrieval,relevance_labels = retrieval(input_ids, segment_ids, input_mask, video, video_mask)
                        loss_s_retrieval,gs_labels = retrieval(generated_input_ids.to(device), generated_segment_ids.to(device), generated_input_mask.to(device), video, video_mask,True,relevance_labels)
                        loss_b_retrieval,_ = retrieval(baseline.to(device),baseline_segments.to(device),baseline_mask.to(device),video,video_mask,True,relevance_labels)
                    
                    
                    

                reward = torch.einsum('nm,n->nm', [log_prob, (cider_scores-args.retrieval_weight*loss_s_retrieval) - (cider_scores_baseline-args.retrieval_weight*loss_b_retrieval)])
                reward *= reward_mask
                reward *= args.reward_multiplier
                reward = torch.sum(reward) / torch.sum(reward_mask)

            #loss = -reward
            loss = (1 - args.gamma_rl) * loss + args.gamma_rl* (-reward)



        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
         



            
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        # else:
        #     loss = loss
        loss.backward()

        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss-cap: %f, Loss-ret: %f,Reward: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.6f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss),
                            float(loss_retrieval),
                            float(reward),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step


# ---------------------------------------->
def get_inst_idx_to_tensor_position_map(inst_idx_list):
    ''' Indicate the position of an instance in a tensor. '''
    return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}


def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
    ''' Collect tensor parts associated to active instances. '''

    _, *d_hs = beamed_tensor.size()
    n_curr_active_inst = len(curr_active_inst_idx)
    new_shape = (n_curr_active_inst * n_bm, *d_hs)

    beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
    beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
    beamed_tensor = beamed_tensor.view(*new_shape)

    return beamed_tensor


def collate_active_info(input_tuples, inst_idx_to_position_map, active_inst_idx_list, n_bm, device):
    assert isinstance(input_tuples, tuple)
    visual_output_rpt, video_mask_rpt = input_tuples

    # Sentences which are still active are collected,
    # so the decoder will not run on completed sentences.
    n_prev_active_inst = len(inst_idx_to_position_map)
    active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
    active_inst_idx = torch.LongTensor(active_inst_idx).to(device)

    active_visual_output_rpt = collect_active_part(visual_output_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_video_mask_rpt = collect_active_part(video_mask_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

    return (active_visual_output_rpt, active_video_mask_rpt), \
           active_inst_idx_to_position_map

def beam_decode_step(decoder, inst_dec_beams, len_dec_seq,
                     inst_idx_to_position_map, n_bm, device, input_tuples, decoder_length=None):

    assert isinstance(input_tuples, tuple)

    ''' Decode and update beam status, and then return active beam idx'''
    def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
        dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
        dec_partial_seq = torch.stack(dec_partial_seq).to(device)
        dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
        return dec_partial_seq

    def predict_word(next_decoder_ids, n_active_inst, n_bm, device, input_tuples):
        visual_output_rpt, video_mask_rpt = input_tuples
        next_decoder_mask = torch.ones(next_decoder_ids.size(), dtype=torch.uint8).to(device)

        dec_output = decoder(visual_output_rpt, video_mask_rpt, next_decoder_ids, next_decoder_mask, shaped=True, get_logits=True)
        dec_output = dec_output[:, -1, :]
        word_prob = torch.nn.functional.log_softmax(dec_output, dim=1)
        word_prob = word_prob.view(n_active_inst, n_bm, -1)
        return word_prob

    def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map, decoder_length=None):
        active_inst_idx_list = []
        for inst_idx, inst_position in inst_idx_to_position_map.items():
            if decoder_length is None:
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
            else:
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position], word_length=decoder_length[inst_idx])
            if not is_inst_complete:
                active_inst_idx_list += [inst_idx]

        return active_inst_idx_list

    n_active_inst = len(inst_idx_to_position_map)
    dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
    word_prob = predict_word(dec_seq, n_active_inst, n_bm, device, input_tuples)

    # Update the beam with predicted word prob information and collect incomplete instances
    active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map,
                                                        decoder_length=decoder_length)

    return active_inst_idx_list

def collect_hypothesis_and_scores(inst_dec_beams, n_best):
    all_hyp, all_scores = [], []
    for inst_idx in range(len(inst_dec_beams)):
        scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
        all_scores += [scores[:n_best]]

        hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
        all_hyp += [hyps]
    return all_hyp, all_scores
# >----------------------------------------

def eval_epoch(args, model,test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=None, test_set=None,retrieval=None,ret=False):
    if hasattr(model, 'module'):
        model = model.module.to(device)

    if retrieval:
        if hasattr(retrieval, 'module'):
            retrieval = retrieval.module.to(device)
        else:
            retrieval = retrieval.to(device)
        retrieval.eval()

    all_result_lists = []
    all_caption_lists = []
    batch_list_t = []
    batch_list_v = []
    batch_sequence_output_list, batch_visual_output_list = [], []
    total_video_num = 0
    if isUniVL:
        cliptokenizer = tokenizer
    else:
        cliptokenizer = ClipTokenizer()
    model.eval()
    
    for batch in tqdm(test_dataloader, desc='validation'):
        batch = tuple(t.to(device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, video, video_mask, \
        pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids = batch

        with torch.no_grad():
            visual_output = model.get_visual_output(video, video_mask)
            # -- Repeat data for beam search
            n_bm = 5 # beam_size
            device = visual_output.device
            n_inst, len_v, v_h = visual_output.size()

            decoder = model.decoder_caption

            # Note: shaped first, then decoder need the parameter shaped=True
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            visual_output_rpt = visual_output.repeat(1, n_bm, 1).view(n_inst * n_bm, len_v, v_h)
            video_mask_rpt = video_mask.repeat(1, n_bm).view(n_inst * n_bm, len_v)
            # -- Prepare beams
            inst_dec_beams = [Beam(n_bm, device=device, tokenizer=tokenizer) for _ in range(n_inst)]
            # -- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
            # -- Decode
            for len_dec_seq in range(1, args.max_words + 1):
                active_inst_idx_list = beam_decode_step(decoder, inst_dec_beams,
                                                        len_dec_seq, inst_idx_to_position_map, n_bm, device,
                                                        (visual_output_rpt, video_mask_rpt))

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                (visual_output_rpt, video_mask_rpt), \
                inst_idx_to_position_map = collate_active_info((visual_output_rpt, video_mask_rpt),
                                                               inst_idx_to_position_map, active_inst_idx_list, n_bm, device)

            batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1)
            result_list = [batch_hyp[i][0] for i in range(n_inst)]

            pairs_output_caption_ids = pairs_output_caption_ids.view(-1, pairs_output_caption_ids.shape[-1])
            caption_list = pairs_output_caption_ids.cpu().detach().numpy()

            k=1
            #print(len(result_list))
            pairs_text = np.zeros((len(result_list),k, args.max_words), dtype=np.long)
            pairs_mask = np.zeros((len(result_list),k, args.max_words), dtype=np.long)
            pairs_segment = np.zeros((len(result_list),k, args.max_words), dtype=np.long)

            for re_idx, re_list in enumerate(result_list):
                decode_text_list = tokenizer.convert_ids_to_tokens(re_list)
                
                if "[SEP]" in decode_text_list:
                    SEP_index = decode_text_list.index("[SEP]")
                    decode_text_list = decode_text_list[:SEP_index]
                if "[PAD]" in decode_text_list:
                    PAD_index = decode_text_list.index("[PAD]")
                    decode_text_list = decode_text_list[:PAD_index]
                decode_text = ' '.join(decode_text_list)
                decode_text = decode_text.replace(" ##", "").strip("##").strip()
                #decode_text = decode_text + " with a"
                all_result_lists.append(decode_text)

                if args.do_eval or ret:
                    words = cliptokenizer.tokenize(decode_text)
                    words = ["[CLS]"] + words
                    total_length_with_CLS = args.max_words - 1
                    if len(words) > total_length_with_CLS:
                        words = words[:total_length_with_CLS]
                    words = words + ["[SEP]"]
                    input_ids = cliptokenizer.convert_tokens_to_ids(words)
                    input_mask = [1] * len(input_ids)
                    segment_ids = [0] * len(input_ids)
                    while len(input_ids) < args.max_words:
                        input_ids.append(0)
                        input_mask.append(0)
                        segment_ids.append(0)
                    assert len(input_ids) == args.max_words
                    assert len(input_mask) == args.max_words
                    assert len(segment_ids) == args.max_words

                    pairs_text[re_idx] = np.array(input_ids)
                    pairs_mask[re_idx] = np.array(input_mask)
                    pairs_segment[re_idx] = np.array(segment_ids)
            if args.do_eval or ret:
                pairs_text = torch.from_numpy(pairs_text).to(device)
                pairs_mask = torch.from_numpy(pairs_mask).to(device)
                pairs_segment = torch.from_numpy(pairs_segment).to(device)
                sequence_output, visual_output = retrieval.get_sequence_visual_output(pairs_text, pairs_segment, pairs_mask, video, video_mask)

                batch_sequence_output_list.append(sequence_output)
                batch_list_t.append((pairs_mask, pairs_segment,))

                batch_visual_output_list.append(visual_output)
                batch_list_v.append((video_mask,))
                

            for re_idx, re_list in enumerate(caption_list):
                decode_text_list = tokenizer.convert_ids_to_tokens(re_list)
                
                if "[SEP]" in decode_text_list:
                    SEP_index = decode_text_list.index("[SEP]")
                    decode_text_list = decode_text_list[:SEP_index]
                if "[PAD]" in decode_text_list:
                    PAD_index = decode_text_list.index("[PAD]")
                    decode_text_list = decode_text_list[:PAD_index]
                decode_text = ' '.join(decode_text_list)
                decode_text = decode_text.replace(" ##", "").strip("##").strip()
                #

                all_caption_lists.append(decode_text)

    # Save full results
    if test_set is not None and hasattr(test_set, 'iter2video_pairs_dict'):
        hyp_path = os.path.join(args.output_dir, "hyp_complete_results.txt")
        with open(hyp_path, "w", encoding='utf-8') as writer:
            writer.write("{}\t{}\t{}\n".format("video_id", "start_time", "caption"))
            for idx, pre_txt in enumerate(all_result_lists):
                video_id, sub_id = test_set.iter2video_pairs_dict[idx]
                start_time = test_set.data_dict[video_id]['start'][sub_id]
                writer.write("{}\t{}\t{}\n".format(video_id, start_time, pre_txt))
        logger.info("File of complete results is saved in {}".format(hyp_path))

    # Save pure results
    print(len(all_result_lists))
    hyp_path = os.path.join(args.output_dir, "hyp.txt")
    with open(hyp_path, "w", encoding='utf-8') as writer:
        for pre_txt in all_result_lists:
            writer.write(pre_txt+"\n")

    ref_path = os.path.join(args.output_dir, "ref.txt")
    with open(ref_path, "w", encoding='utf-8') as writer:
        for ground_txt in all_caption_lists:
            writer.write(ground_txt + "\n")

    if args.datatype == "msrvtt" or args.datatype=="msvd":
        all_caption_lists = []
        sentences_dict = test_dataloader.dataset.sentences_dict
        video_sentences_dict = test_dataloader.dataset.video_sentences_dict
        for idx in range(len(sentences_dict)):
            video_id, _ = sentences_dict[idx]
            sentences = video_sentences_dict[video_id]
            all_caption_lists.append(sentences)
        if args.datatype != "msvd": # if number of caption for each video is different, use this
            all_caption_lists = [list(itms) for itms in zip(*all_caption_lists)]
    else:
        all_caption_lists = [all_caption_lists]

    # retrieval evaluation

    


    if args.do_eval or ret:
        
        sim_matrix =run_on_single_gpu(retrieval, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list)
        #if not isUniVL:
        sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)


        logger.warning("retrieval evaluation ")
        logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        tv_metrics = compute_metrics(sim_matrix)
        vt_metrics = compute_metrics(sim_matrix.T)
        logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))

        logger.info("Text-to-Video:")
        logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                    format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
        logger.info("Video-to-Text:")
        logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                    format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))


                
    # Evaluate
    if args.datatype == "msvd":
        all_result_dict = {}
        all_caption_dict = {}
        for i in range(len(all_result_lists)):
            all_result_dict[i] = [all_result_lists[i]]
        for i in range(len(all_caption_lists)):
            all_caption_dict[i]=all_caption_lists[i]
        # Evaluate
        # for i in range(len(all_result_lists)):
        #     all_result_dict[i] = [all_caption_lists[i][1]]
        metrics_nlg = score(all_caption_dict,all_result_dict)
    else:
        metrics_nlg = nlgEvalObj.compute_metrics(ref_list=all_caption_lists, hyp_list=all_result_lists)
    logger.info(">>>  BLEU_1: {:.4f}, BLEU_2: {:.4f}, BLEU_3: {:.4f}, BLEU_4: {:.4f}".
                format(metrics_nlg["Bleu_1"], metrics_nlg["Bleu_2"], metrics_nlg["Bleu_3"], metrics_nlg["Bleu_4"]))
    logger.info(">>>  METEOR: {:.4f}, ROUGE_L: {:.4f}, CIDEr: {:.4f}".format(metrics_nlg["METEOR"], metrics_nlg["ROUGE_L"], metrics_nlg["CIDEr"]))
    #logger.info(">>> SPICE: {:.4f}".format(metrics_nlg["SPICE"]))

    return metrics_nlg

def main(args,tokenizer,cliptokenizer):
    global logger

    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    model = init_model(args, device, n_gpu, args.local_rank)

    
    retrieval = load_model_retrieval(-1, args, n_gpu, device, model_file=args.model_file_retrieval)
    eval_epoch_retrieval(args, retrieval, test_dataloader, device, n_gpu,logger)

    nlgEvalObj = NLGEval(no_overlap=False, no_skipthoughts=True, no_glove=True, metrics_to_omit=None)


    
    if args.local_rank == 0:
        logger.info("***** Running val *****") 
        logger.info("  Num examples = %d", val_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(val_dataloader))
        logger.info("***** Running test *****") 
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))


    if args.do_train:

        retrieval.eval()
        
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        coef_lr = args.coef_lr
        if args.init_model:
            coef_lr = 1.0
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)

        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        best_score = {"Bleu_4": 0.00001, "METEOR": 0.00001, "CIDEr": 0.00001, "ROUGE_L": 0.00001}
        best_output_model_file = {"Bleu_4": None, "METEOR": None, "CIDEr": None, "ROUGE_L": None}
        assert args.target_metric in best_score.keys()
        assert args.patience_metric in best_score.keys()
        global_step = 0
        stop_signal = torch.zeros(2).cuda() # index 0 for early stopping based on num of epoch, index 1 for low metrics
        for epoch in tqdm(range(args.epochs)):
            train_sampler.set_epoch(epoch)

            tr_loss, global_step = train_epoch(epoch, args, model, retrieval, train_dataloader, tokenizer, cliptokenizer,device, n_gpu, optimizer,
                                               scheduler, global_step, nlgEvalObj=nlgEvalObj, local_rank=args.local_rank)

            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
                #output_model_file = save_model(epoch, args, model, type_name="")
                if epoch >= 10:
                    metric_scores = eval_epoch(args, model, val_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj,retrieval=retrieval)
                    for met in best_score.keys():
                        if met == args.target_metric:
                            if metric_scores[met] <= 0.001:
                                logger.warning("One of the metrics is less than 0.001. The training will be stopped.")
                                stop_signal[1] = 1
                                break
                            if best_score[met] <= metric_scores[met]:
                                output_model_file = save_model(epoch, args, model, type_name="")
                                best_score[met] = metric_scores[met]
                                best_output_model_file[met] = output_model_file
                                if met==args.patience_metric:
                                    stop_signal[0] = 0
                            else:
                                if met==args.patience_metric:
                                    stop_signal[0] += 1
                            logger.info("The best model based on {} is: {}, the {} is: {:.4f}".format(met, best_output_model_file[met], met, best_score[met]))

                    for gp in range(1, args.n_gpu): # TODO, try to use broadcast function of pytorch
                        torch.distributed.send(stop_signal, dst=gp)

                    if stop_signal[0]>=args.patience:
                        logger.warning("Early stopping, no improvement after {} epochs at local rank {}".format(args.patience, args.local_rank))
                        break
                    if stop_signal[1] == 1:
                        break
                else:
                    for gp in range(1, args.n_gpu): # TODO, try to use broadcast function of pytorch
                        torch.distributed.send(stop_signal, dst=gp)
                    logger.warning("Skip the evaluation after {}-th epoch.".format(epoch+1))
            else:
                torch.distributed.recv(stop_signal, src=0)
                if stop_signal[0]>=args.patience:
                    logger.warning("Early stopping, no improvement after {} epochs at local rank {}".format(args.patience, args.local_rank))
                    break
                elif stop_signal[1]==1:
                    logger.warning("One of the metrics is less than 0.001")
                    break

        if args.local_rank == 0:
            test_scores = {}
            for met in best_score.keys():
                if met == args.target_metric:
                    model = None
                    model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file[met])
                    retrieval = None
                    retrieval = load_model_retrieval(-1, args, n_gpu, device, model_file=args.model_file_retrieval)
                    
                    metric_scores = eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj,retrieval=retrieval,ret=True)
                    test_scores[met] = metric_scores
            for met in test_scores.keys():
                logger.info("Test score based on the best {} model ({}) : {}".format(met, best_output_model_file[met], str(test_scores[met])))
    elif args.do_eval:
        if args.local_rank == 0:
            model = None
            if args.eval_type =='captioning':
            
                model = load_model(-1, args, n_gpu, device, model_file=args.model_file)
                eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj,retrieval=retrieval)
            if args.eval_type =='retrieval':
                model = load_model_retrieval(-1, args, n_gpu, device, model_file=args.model_file_retrieval)
                eval_epoch_retrieval(args, model, test_dataloader, device, n_gpu,logger,retrieval=retrieval)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = get_args()
    args = set_seed_logger(args)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    

    if isUniVL:
        rettokenizer = tokenizer
    else:
        rettokenizer = ClipTokenizer()
    
    ## ####################################
    # dataloader loading
    ## ####################################
    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None \
            or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer,rettokenizer,"test")

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer,rettokenizer,"val")
    else:
        val_dataloader, val_length = test_dataloader, test_length

    ## report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length
    train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer,rettokenizer)
    



    main(args,tokenizer,rettokenizer)
