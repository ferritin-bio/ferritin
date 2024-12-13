from transformers import AutoModel
from transformers import AutoTokenizer

model_repo = "facebook/esm2_t6_8M_UR50D"
model = AutoModel.from_pretrained(model_repo, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)

print(model, tokenizer)


# EsmTokenizer(name_or_path='facebook/esm2_t6_8M_UR50D', vocab_size=33, model_max_length=1000000000000000019884624838656, is_fast=False, padding_side='right', truncation_side='right', special_tokens={'eos_token': '<eos>', 'unk_token': '<unk>', 'pad_token': '<pad>', 'cls_token': '<cls>', 'mask_token': '<mask>'}, clean_up_tokenization_spaces=False, added_tokens_decoder={
# 	0: AddedToken("<cls>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
# 	1: AddedToken("<pad>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
# 	2: AddedToken("<eos>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
# 	3: AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
# 	32: AddedToken("<mask>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
# }
# )
#
#
# tokenizer._id_to_token
# Out[5]:
# {0: '<cls>',
#  1: '<pad>',
#  2: '<eos>',
#  3: '<unk>',
#  4: 'L',
#  5: 'A',
#  6: 'G',
#  7: 'V',
#  8: 'S',
#  9: 'E',
#  10: 'R',
#  11: 'T',
#  12: 'I',
#  13: 'D',
#  14: 'P',
#  15: 'K',
#  16: 'Q',
#  17: 'N',
#  18: 'F',
#  19: 'Y',
#  20: 'M',
#  21: 'H',
#  22: 'W',
#  23: 'C',
#  24: 'X',
#  25: 'B',
#  26: 'U',
#  27: 'Z',
#  28: 'O',
#  29: '.',
#  30: '-',
#  31: '<null_1>',
#  32: '<mask>'}
