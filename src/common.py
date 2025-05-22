from datasets import load_dataset, concatenate_datasets

def getDataset(opt_lang):
    lang_list = ("afr", "xho", "zul", "ven", "tso", "tsn", "ssw", "nso", "sot")

    if opt_lang in lang_list:
        datasets = load_dataset(f"danielshaps/nchlt_speech_{opt_lang}")
        return concatenate_datasets([split for split in datasets.values()])
    else:
        raise ValueError(f"Invalid `opt_lang`: {opt_lang}")