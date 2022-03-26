# easy_align
make align torch weights to paddle easy.

## 安装和卸载

```shell
git clone https://github.com/MiuGod0126/easy_align
cd easy_align
# 安装 
python setup.py install
# 卸载
pip uninstall easyAlign
```

## 创建转换器

```python
import torch
import paddle
from easyAlign import Torch2PaddleConverter
# 加载权重
torch_weights=torch.load('checkpoint_best.pt')['model']
paddle_weights=paddle.load('transformer.pdparams')

# 创建转换器
converter=Torch2PaddleConverter(torch_weights,paddle_weights)
```

## 写转换规则

使用compare_keys可以规整的打印torch和paddle的参数，便于写转换规则。

```python
# 打印参数
converter.compare_keys()
'''
torch_key and shape:                                                torch_key and shape:
-------------------------------------------[1]-------------------------------------------
encoder.version [1]                 src_word_embedding.word_embedding.weight [20557, 512]        	
-------------------------------------------[2]-------------------------------------------
encoder.embed_tokens.weight [24972, 512]   src_pos_embedding.pos_encoder.weight [257,512]
...
'''

```

需要有skip_weights，donot_transpose，torch_to_paddle_keys，special_case_fn（可以没有）。

```python

## 跳过的权重
skip_weights=["encoder.version","decoder.version","encoder.embed_positions._float_tensor","decoder.embed_positions._float_tensor"] 
donot_transpose=['encoder.embed_tokens.weight','decoder.embed_tokens.weight'] # 不需要转置的weight，如embed
## 参数名名映射 
torch_to_paddle_keys={"encoder.embed_tokens.weight":"src_word_embedding.word_embedding.weight",
                     "decoder.embed_tokens.weight":"trg_word_embedding.word_embedding.weight",
                     "fc1":"linear1",
                     "fc2":"linear2",
                      "self_attn_layer_norm":"norm1",
                      "encoder_attn_layer_norm":"norm2",
                      "encoder_attn.":"cross_attn.", # 注意别把norm的替换掉了
                       "decoder.output_projection.weight":"linear.weight"
                     }
# 特殊参数名处理
def special_case_fn(key):
    special_key="final_layer_norm" # 在encoder是norm2 在decoder是norm3
    special_val="norm2" if paddle_k.find("encoder")!=-1 else "norm3"
    key=key.replace(special_key,special_val)
    return key
```

## 转换权重

```python
# 转换权重
paddle_weights=converter.align_weights(skip_weights,donot_transpose,torch_to_paddle_keys,special_case_fn)
paddle.save(paddle_weights,'model.pdparams')
```

