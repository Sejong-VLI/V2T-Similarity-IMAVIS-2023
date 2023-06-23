import torch
from torch.utils.data import DataLoader
# from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_DataLoader
# from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_TrainDataLoader

from <VIDEOCAPTIONINGFOLDER>.dataloaders.dataloader_msrvtt_caption import MSRVTT_Caption_DataLoader
from <VIDEOCAPTIONINGFOLDER>.dataloaders.dataloader_msvd import MSVD_Loader

from torch.utils.data import (SequentialSampler)


def dataloader_msvd_val_test(args, tokenizer, cliptokenizer,split_type="val",):
    msvd = MSVD_Loader(
        data_path=args.data_path,
        inference_path = None,
        features_path=args.features_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        cliptokenizer=cliptokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
        use_raw_video=False,
    )

    sampler = SequentialSampler(msvd)
    dataloader_msvd = DataLoader(
        msvd,
        sampler=sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=True,
        drop_last=False,
        # timeout=10,
    )
    return dataloader_msvd, len(msvd)

def dataloader_msvd(args, tokenizer, cliptokenizer,split_type="train",):
    msvd = MSVD_Loader(
        data_path=args.data_path,
        inference_path = None,
        features_path=args.features_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        cliptokenizer=cliptokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
        use_raw_video=False,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msvd)
    dataloader_msvd = DataLoader(
        msvd,
        sampler=train_sampler,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=True,
        #  timeout=10,
        shuffle=(train_sampler is None),
        drop_last=True,)
    return dataloader_msvd, len(msvd), train_sampler


def dataloader_msrvtt_train(args, tokenizer,cliptokenizer):
    msrvtt_dataset = MSRVTT_Caption_DataLoader(
       
        json_path=args.data_path,
        features_path=args.features_path,
        inference_path=None,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        cliptokenizer=cliptokenizer,
        max_frames=args.max_frames,
        use_raw_video=False,
       
       


    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler

def dataloader_msrvtt_val_test(args, tokenizer, cliptokenizer,split_type="test"):
    msrvtt_testset = MSRVTT_Caption_DataLoader(
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        inference_path=None,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        cliptokenizer=cliptokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
        use_raw_video=False,

    )
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)





DATALOADER_DICT = {}
DATALOADER_DICT["msrvtt"] = {"train":dataloader_msrvtt_train, "val":dataloader_msrvtt_val_test, "test":dataloader_msrvtt_val_test}
DATALOADER_DICT["msvd"] = {"train":dataloader_msvd, "val":dataloader_msvd_val_test, "test":dataloader_msvd_val_test}
