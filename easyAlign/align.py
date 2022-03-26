class Torch2PaddleConverter():
    def __init__(self, torch_weights, paddle_weights):
        self.torch_keys = list(torch_weights.keys())
        self.torch_weights = torch_weights
        self.paddle_keys = list(paddle_weights.keys())
        self.paddle_weights = paddle_weights

    def compare_keys(self):
        ''' print parameters info of torch and paddle clearly.'''
        idx = 0
        print("{0:<60} \t {0:>60}".format("torch_key and shape:", "paddle_key and shape:"))
        for (torch_k, torch_w), (paddle_k, paddle_v) in zip(self.torch_weights.items(), self.paddle_weights.items()):
            print("-" * 60 + f"[{idx + 1}]" + "-" * 60)
            msg_torch = f"{torch_k} {list(torch_w.shape)}"
            msg_paddle = f"{paddle_k} {list(paddle_v.shape)}"
            print("{0:<60} \t {1:>60}".format(msg_torch, msg_paddle))
            idx += 1
        torch_len, paddle_len = len(self.torch_keys), len(self.paddle_keys)
        if torch_len > paddle_len:
            for i in range(idx, torch_len):
                print("-" * 60 + f"[{i + 1}]" + "-" * 60)
                torch_k = self.torch_keys[i]
                torch_w = self.torch_weights[torch_k]
                msg_torch = f"{torch_k} {list(torch_w.shape)}"
                print("{0:<60} \t ".format(msg_torch))
        elif torch_len < paddle_len:
            for i in range(idx, paddle_len):
                print("-" * 60 + f"[{i + 1}]" + "-" * 60)
                paddle_k = self.paddle_keys[i]
                paddle_w = self.paddle_weights[paddle_k]
                msg_paddle = f"{paddle_k} {list(paddle_w.shape)}"
                print("{0:>60} \t ".format(msg_paddle))

    def align_weights(self, skip_weights, donot_transpose, torch_to_paddle_keys, special_case_fn=None):
        '''
            Args:
                skip_weights: list[str],skip some unnecessary torch parameter names.
                donot_transpose: list[str], skip some 2dim weight when transpose, such as embedding.
                torch_to_paddle_keys: dict[str:str],rules for transferring  torch key to paddle key.
                special_case_fn: function to process some special key map.
            return:
                aligned paddle weights.
        '''
        for i, (torch_k, torch_w) in enumerate(self.torch_weights.items()):
            transpose = False
            # 1.跳过多余的
            if torch_k in skip_weights:
                continue
            # 2.对二维需要转置的weight转置（linear，非embed）
            if torch_k.find('.weight') != -1:
                if torch_k not in donot_transpose:  # 排除embed
                    if torch_w.ndim == 2:  # 排除norm
                        torch_w = torch_w.transpose(0, 1)
                        transpose = True
            # 3.参数名映射
            paddle_k = torch_k
            for k, v in torch_to_paddle_keys.items():
                paddle_k = paddle_k.replace(k, v)
            # 特殊key处理
            if special_case_fn is not None:
                paddle_k = special_case_fn(paddle_k)
            print(f"Converting [{i + 1}]: {torch_k} => {paddle_k} | is_transpose {transpose}")
            # 4.存入paddle权重
            self.paddle_weights[paddle_k] = torch_w.cpu().detach().numpy()
        print('aligned all parameters.')
        return self.paddle_weights
