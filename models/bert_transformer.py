from transformers import BertModel
from transformers.models.bert.modeling_bert import BertConfig
import torch
from transformers import (
    DataCollatorForLanguageModeling,
    BertTokenizer,
)
import numpy as np

class BertEmbedder:
    #from https://gist.github.com/shubhamagarwal92/37ccb747f7130a35a8e76aa66d60e014
    def __init__(self,
                 pretrained_weights='bert-base-uncased',
                #  tokenizer_class=BertTokenizer,
                 model_class=BertModel,
                 max_seq_len=8,
                 config=None):
        super().__init__()
        self.pretrained_weights = pretrained_weights
        # self.tokenizer_class = tokenizer_class
        self.model_class = model_class
        # self.tokenizer = self.tokenizer_class.from_pretrained(pretrained_weights)
        self.bert_config = BertConfig(
            vocab_size=config["vocab_size"],#len(self.num_objs)+len(self.num_attrs),#len(dset.pairs),#self.config["vocab_size"],#self.num_objs,#
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )
        self.model = self.model_class.from_pretrained(pretrained_weights).to('cuda')
        self.max_seq_len = max_seq_len
        # tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        # model = BertModel.from_pretrained(pretrained_weights)
        tokenizer = "bert-base-uncased"
        self.tokenizer = self.get_pretrained_tokenizer(tokenizer)
        self.vocab_size = self.tokenizer.vocab_size
        
    def get_pretrained_tokenizer(self,from_pretrained):
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                BertTokenizer.from_pretrained(
                    from_pretrained, do_lower_case="uncased" in from_pretrained
                )
            torch.distributed.barrier()
        return BertTokenizer.from_pretrained(
            from_pretrained, do_lower_case="uncased" in from_pretrained
        )

    def get_text(self, text):
        #index, caption_index = self.index_mapper[raw_index]

        # text = self.all_texts[raw_index]#[caption_index]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_special_tokens_mask=True,
        )

        return (text, encoding)

    def get_bert_embeddings(self,
                            input_ids,mask):
        # examples = create_examples(raw_text)

        # features = convert_examples_to_features(
        #     examples, self.tokenizer, self.max_seq_len, True)

        # all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        # all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        # all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        input_ids=torch.from_numpy(np.asarray(input_ids)).unsqueeze(0).to('cuda')
        mask=torch.from_numpy(np.asarray(mask)).unsqueeze(0).to('cuda')
        last_hidden_states = self.model(input_ids,attention_mask = mask).last_hidden_state  # Models outputs are now tuples
        # import pdb; pdb.set_trace()
        return last_hidden_states[:,0,:]