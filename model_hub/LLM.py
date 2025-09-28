import time
import torch
from termcolor import colored


class LLM:
    """
    A class representing the LLM (currently support Llama and Qwen).
    """

    def __init__(
        self, 
        model_name: str,
        max_length: int,
        dtype: torch.dtype,
        device_map: str
    ) -> None:
        """ Initializes the LLM.
        Args:
            model_name (str): The name of the model.
            max_length (int): The maximum length (prefill+decode) of sequences.
            dtype (torch.dtype): The data type for model computations.
            device_map (str): The device for model, suppor 'cuda:x' or 'auto (automatically use all visible GPUs)'.
        """

        self.model_name = model_name
        self.max_length = max_length
        self.dtype = dtype
        self.device_map = device_map


    def layer_prefill(self, layer_idx, start_bdx, hidden_states):
        # print(f'Layer = {layer_idx}, start_bdx = {start_bdx}')

        bsz, seq_len, dim = hidden_states.shape
        layer = self.layers[layer_idx]
        
        # original hidden_states used as residual, clone a new one to process
        temp_hidden_states = hidden_states.clone()

        # chunk for lower memory comsumption
        for start_idx in range(0, seq_len, 8192//bsz):
            end_idx = min(seq_len, start_idx + 8192//bsz)
            temp_hidden_states[:, start_idx:end_idx, :] = self.layernorm(temp_hidden_states[:, start_idx:end_idx, :], 
                                                                         layer.input_layernorm_variance_epsilon, 
                                                                         layer.input_layernorm_weight)
        
        query_states, key_states, value_states = self.wqkv(temp_hidden_states, layer)
        del temp_hidden_states
        torch.cuda.empty_cache()
        query_states, key_states = self.position_embedd(query_states, key_states)

        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim)       # reshape [bs, seq_len, dim] => [bs, seq_len, head, head_dim]
        key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        key_states, value_states = self.kv_cache.prefill_update_kv_cache(query_states, key_states, value_states, layer_idx, start_bdx)
        torch.cuda.empty_cache()

        temp_attn_out = self.prefill_attention(query_states, key_states, value_states)

        self.kv_cache.sync(layer_idx, start_bdx)

        del query_states, key_states, value_states
        torch.cuda.empty_cache()

        hidden_states += self.wo(temp_attn_out, layer, temp_attn_out.shape[0], seq_len, dim)
        del temp_attn_out
        torch.cuda.empty_cache()

        # post attention
        residual = hidden_states.clone()

        # chunk for lower memory comsumption
        for start_idx in range(0, seq_len, 8192//bsz):
            end_idx = min(seq_len, start_idx + 8192//bsz)
            hidden_states[:, start_idx:end_idx, :] = self.layernorm(hidden_states[:, start_idx:end_idx, :], 
                                                                    layer.post_attention_layernorm_variance_epsilon, 
                                                                    layer.post_attention_layernorm_weight)
            hidden_states[:, start_idx:end_idx, :] = self.mlp(hidden_states[:, start_idx:end_idx, :], layer)   
        
        hidden_states += residual

        del residual
        torch.cuda.empty_cache()
                                                                                                   
        return hidden_states


    def layer_decode(self, layer_idx, hidden_states):
        # print(f'Layer = {layer_idx}')

        residual = hidden_states
        bsz, seq_len, dim = hidden_states.shape
        layer = self.layers[layer_idx]

        hidden_states = self.layernorm(hidden_states, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
        
        query_states, key_states, value_states = self.wqkv(hidden_states, layer)
        query_states, key_states = self.position_embedd(query_states, key_states)

        query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim)

        key_states, value_states = self.kv_cache.decode_update_kv_cache(key_states, value_states, layer_idx)
        attn_out = self.decode_attention(query_states, key_states, value_states, layer_idx)
        hidden_states = self.wo(attn_out, layer, bsz, seq_len, dim)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layernorm(hidden_states, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
        hidden_states = self.mlp(hidden_states, layer)
        hidden_states = residual + hidden_states

        return hidden_states


    def prefill_forward(self, inputs_ids):
        bsz, seq_len = inputs_ids.shape
        device = inputs_ids.device

        last_hidden_states = torch.empty((bsz, 1, self.hidden_size), dtype=self.dtype, device=device)
        for start_bdx in range(0, bsz, 1):
            end_bdx = min(bsz, start_bdx + 1)
            hidden_states = self.word_embedding(inputs_ids[start_bdx:end_bdx])  # [1, seq_len, hidden_size]

            if self.num_gpus > 1:
                for ldx in range(self.num_layers):
                    hidden_states = self.layer_prefill(ldx, start_bdx, hidden_states)
                    hidden_states = self.parameter_move(hidden_states, ldx)
                    torch.cuda.empty_cache()
                last_hidden_states[start_bdx:end_bdx] = hidden_states[:, -1:, :].to(self.layers[0].device)
            else:
                for ldx in range(self.num_layers):
                    hidden_states = self.layer_prefill(ldx, start_bdx, hidden_states)
                    torch.cuda.empty_cache()
                last_hidden_states[start_bdx:end_bdx] = hidden_states[:, -1:, :]
        
        last_hidden_states = self.layernorm(last_hidden_states.contiguous(), self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(last_hidden_states)
        
        return logits
        

    def decode_forward(self, inputs_ids):
        hidden_states = self.word_embedding(inputs_ids)

        if self.num_gpus > 1:
            for ldx in range(self.num_layers):
                hidden_states = self.layer_decode(ldx, hidden_states)
                hidden_states = self.parameter_move(hidden_states, ldx)
            hidden_states = hidden_states.to(self.layers[0].device)
        else:
            for ldx in range(self.num_layers):
                hidden_states = self.layer_decode(ldx, hidden_states)
        
        hidden_states = self.layernorm(hidden_states[:, -1:, :], self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(hidden_states)
        
        return logits


    def inference(self, inputs_ids, stop_words=None):
        
        outputs_ids = []    # multi iteration, multi request
        output_ids = []     # single iteration, multi request

        processed_stop_words = None
        if stop_words is not None:
            processed_stop_words = []
            for stop_seq in stop_words:
                if isinstance(stop_seq, str):
                    if hasattr(self, 'tokenizer'):
                        processed_stop_words.append(self.tokenizer.encode(stop_seq, add_special_tokens=False))
                    else:
                        raise ValueError("tokenizer does not exist in LLM instance")
                else:
                    processed_stop_words.append(stop_seq)
        print(f"stop word tokens: {processed_stop_words}")

        print("Start prefilling ...")
        torch.cuda.synchronize()
        prefill_start = time.time()

        logits = self.prefill_forward(inputs_ids=inputs_ids)
        output_ids = logits.argmax(dim=-1)
        outputs_ids.append(output_ids)
        self.move()

        torch.cuda.synchronize()
        prefill_end = time.time()
        print(colored(f"Prefilling latency: {round((prefill_end - prefill_start), 4)} s\n", 'green'))

        print("Start decoding ...")
        decode_start = time.time()

        batch_size = output_ids.shape[0]

        finished = [False] * batch_size

        # 记录每个序列遇到stop word时应截断的位置
        stop_pos = [None] * batch_size
        for _ in range(self.max_new_length-1):
            logits = self.decode_forward(inputs_ids=output_ids)
            next_token = logits.argmax(dim=-1)

            # 对于已完成的序列，保持其token不变
            for i in range(batch_size):
                if finished[i]:
                    next_token[i] = output_ids[i]
            outputs_ids.append(next_token)
            output_ids = next_token

            # 检查每个序列是否遇到停止词
            if processed_stop_words is not None:
                generated = torch.cat(outputs_ids, dim=-1)
                for batch_idx in range(batch_size):
                    if finished[batch_idx]:
                        continue
                    for stop_seq in processed_stop_words:
                        if len(stop_seq) == 0:
                            continue
                        if generated.shape[1] >= len(stop_seq):
                            if generated[batch_idx, -len(stop_seq):].tolist() == stop_seq:
                                finished[batch_idx] = True
                                stop_pos[batch_idx] = generated.shape[1] - len(stop_seq)
            # 如果所有序列都完成则break
            if all(finished):
                break

        decode_end = time.time()
        print(colored(
            f"Decoding latency: {round((decode_end - decode_start) * 1000 / (self.max_new_length - 1), 2)} ms/step, "
            f"Throughput: {round(self.batch_size * (self.max_new_length - 1) / (decode_end - decode_start), 2)} tokens/s, "
            f"Decoding total time: {round((decode_end - decode_start), 4)} s\n",
            'green'
        ))

        outputs_ids = torch.cat(outputs_ids, dim=-1)
        # 截断stop word
        if processed_stop_words is not None and any(pos is not None for pos in stop_pos):
            results = []
            for i, row in enumerate(outputs_ids.tolist()):
                if stop_pos[i] is not None:
                    results.append(row[:stop_pos[i]])
                else:
                    results.append(row)
            outputs_ids = results
        else:
            outputs_ids = outputs_ids.tolist()

        return outputs_ids


    def generate(self, attention_type, inputs_ids, attention_masks, max_new_length, attn_config=None):
        """ LLM Inference.
        Args:
            attention_type: str,
            input_ids (torch.tensor): The input of LLM.
            attention_masks (torch.tensor): The attention masks of LLM.
            max_new_length (int): The maximum length of generated sequences.
        """

        bs, input_length = inputs_ids.shape
        assert input_length + max_new_length <= self.max_length, \
        f"Error: input_length({input_length}) + max_new_length({max_new_length}) exceeds max_length({self.max_length})"

        self.batch_size = bs
        self.input_length = input_length
        self.max_new_length = max_new_length
        self.attention_type = attention_type

        valid_start = attention_masks.shape[1] - torch.sum(attention_masks, dim=-1).detach().cpu().numpy()
        del attention_masks
        torch.cuda.empty_cache()

        print("Allocate GPU buffers and CPU pin memory ...\n")
        self.init_kv_cache(input_length, valid_start, attn_config)

        outputs = self.inference(inputs_ids, stop_words=["</s>", "assistant:", "Question:", "user:", "USER:"])
        # stop word tokens: [[524, 82, 29], [78191, 25], [14924, 25], [882, 25], [6584, 25]]

        return outputs
