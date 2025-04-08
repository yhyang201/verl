import torch


from verl import DataProto
from verl.trainer.ppo.core_algos import compute_policy_loss, kl_penalty, agg_loss
from verl.utils.py_functional import append_to_dict

from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx

import verl.utils.torch_functional as verl_F

def compute_sppo_loss(
    old_log_prob: torch.Tensor,      # (bs, seq_len)
    log_prob: torch.Tensor,          # (bs, seq_len)
    rewards: torch.Tensor,        # (bs,)
    response_mask: torch.Tensor,     # (bs, seq_len)
    eta: float = 1.0,
    loss_agg_mode: str = "token-mean"
):
    """
    SPPO Loss computation.
    """
    # Compute log-ratios over masked tokens
    log_prob_sum = (log_prob * response_mask).sum(dim=1)  # (bs,)
    old_log_prob_sum = (old_log_prob * response_mask).sum(dim=1)  # (bs,)
    log_ratios = log_prob_sum - old_log_prob_sum  # (bs,)

    preference = eta * (response_mask - 0.5)  # (bs,)
    loss_vec = (log_ratios - rewards) ** 2  # (bs,)
    

    if loss_agg_mode == "seq-mean-token-sum":
        loss = loss_vec.mean()
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_lengths = response_mask.sum(dim=1)  # (bs,)
        token_mean_loss = loss_vec / seq_lengths.clamp(min=1)
        loss = token_mean_loss.mean()
    elif loss_agg_mode == "token-mean":
        sample_mask = response_mask.any(dim=1).float()  # (bs,)
        loss = verl_F.masked_mean(loss_vec, sample_mask)       
    else:
        raise ValueError(f"Unsupported loss_agg_mode: {loss_agg_mode}")

    return loss, log_ratios, preference


def update_policy(self, data: DataProto):
    # make sure we are in training mode
    self.actor_module.train()

    temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

    select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'token_level_rewards']
    if self.config.use_kl_loss:
        select_keys.append('ref_log_prob')
    batch = data.select(batch_keys=select_keys).batch
    has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

    # Split to make minibatch iterator for updating the actor
    # See PPO paper for details. https://arxiv.org/abs/1707.06347
    if has_multi_modal_inputs:
        num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
        non_tensor_select_keys = ['multi_modal_inputs']
        dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
    else:
        dataloader = batch.split(self.config.ppo_mini_batch_size)

    metrics = {}
    for epoch in range(self.config.ppo_epochs):
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if has_multi_modal_inputs:
                self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
            elif self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

            self.actor_optimizer.zero_grad()

            for data in micro_batches:
                # Support all hardwares
                if isinstance(data, DataProto):
                    data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                else:
                    data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload
                responses = data['responses']
                response_length = responses.size(1)
                attention_mask = data['attention_mask']
                response_mask = attention_mask[:, -response_length:]
                old_log_prob = data['old_log_probs']
                rewards = data['token_level_rewards']      
                          
                entropy_coeff = self.config.entropy_coeff
                loss_agg_mode = self.config.loss_agg_mode

                # all return: (bsz, response_length)
                entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)
                print(old_log_prob.device, log_prob.device, rewards.device)
                pg_loss, log_ratios, preference = compute_sppo_loss(
                    old_log_prob=old_log_prob,
                    log_prob=log_prob,
                    rewards=rewards.sum(axis=-1),
                    response_mask=response_mask,
                    eta=self.config.get('sppo_eta', 1.0),
                    loss_agg_mode=loss_agg_mode,
                )
                print(pg_loss.device, log_ratios.device, preference.device)
                # compute entropy loss from entropy
                entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                # compute policy loss
                policy_loss = pg_loss - entropy_loss * entropy_coeff

                if self.config.use_kl_loss:
                    ref_log_prob = data['ref_log_prob']
                    # compute kl loss
                    kld = kl_penalty(logprob=log_prob,
                                        ref_logprob=ref_log_prob,
                                        kl_penalty=self.config.kl_loss_type)
                    kl_loss = agg_loss(loss_mat=kld,
                                        loss_mask=response_mask,
                                        loss_agg_mode=self.config.loss_agg_mode)

                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metrics['actor/kl_loss'] = kl_loss.detach().item()
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef

                if self.config.use_dynamic_bsz:
                    # relative to the dynamic bsz
                    loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                else:
                    loss = policy_loss / self.gradient_accumulation
                loss.backward()

                data_metrics = {
                    'actor/entropy': entropy_loss.detach().item(),
                    'actor/pg_loss': pg_loss.detach().item(),
                    'actor/log_ratio_mean': log_ratios.mean().detach().item(),
                    'actor/preference_mean': preference.mean().detach().item(),
                }
                append_to_dict(metrics, data)

            grad_norm = self._optimizer_step()
            data = {'actor/grad_norm': grad_norm.detach().item()}
        append_to_dict(metrics, data)
    self.actor_optimizer.zero_grad()
    return metrics
