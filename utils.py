# 20种多肽缩写字符
sequences = 'ACDEFGHIKLMNPQRSTVWY'

# 获取全部字符（氨基酸）集合
all_chars = sorted(set(''.join(sequences)))
vocab_size = len(all_chars)

# 建立字符与索引的映射
char2idx = {ch: i for i, ch in enumerate(all_chars)}
idx2char = {i: ch for ch, i in char2idx.items()}