from sacred import Experiment

ex = Experiment("ViLT")


def _loss_names(d):
    ret = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 0,
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "vilt"
    seed = 0
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    # batch_size = 72  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    image_size = 384
    max_image_len = -1
    patch_size = 32
    draw_false_image = 1
    image_only = False

    # Text Setting
    vqav2_label_size = 3129
    max_text_len = 8
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15#0.15
    draw_false_text = 0

    # Transformer Setting
    vit = "vit_base_patch16_224"
    hidden_size = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"

    # Downstream Setting
    get_recall_metric = False

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False

    # below params varies with the environment
    data_root = '/mit-state'
    log_dir = '/OADis/log'
    per_gpu_batchsize = 32  #you should define this manually with per_gpu_batch_size=#
    cfg='config/mit-states.yml'
    wb_name='abc'
    num_freeze_layers=6
    num_gpus = 4
    num_nodes = 1
    load_path = ""
    num_workers = 0
    precision = 16
    lr=1e-3
    lr_transformer=1e-4
    lr_cross=1e-1
    k=5
    offset_val=0.2
    neta=0.05

