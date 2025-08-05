# create data
python thirdparty/verl/sf_scripts/skill_factory_data.py \
    --hf_repo TAUR-dev/D-DATA-canonical_dataset_splits-v1-7_13_25 \
    --output_folder /scratch/yl11330/skill-factory/data/countdown_3arg

# run ppo (1 node 2 gpu), countdown_3arg
# NOTE: change trainer.project_name to log into your own wandb project
export RAY_DEBUG=legacy && unset ROCR_VISIBLE_DEVICES && CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    trainer.logger=['console','wandb'] \
    trainer.project_name=jackrl \
    trainer.experiment_name=test \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=10 \
    trainer.total_epochs=100 \
    data.train_files=/scratch/yl11330/skill-factory/data/countdown_3arg/train.parquet \
    data.val_files=/scratch/yl11330/skill-factory/data/countdown_3arg/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=TAUR-dev/SIE-countdown3arg_sft1ep_1e6lr_mix_ss_pse_vote_ansrev-sft \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    critic.model.path=TAUR-dev/SIE-countdown3arg_sft1ep_1e6lr_mix_ss_pse_vote_ansrev-sft \
    critic.optim.lr=5e-5 \
    critic.ppo_micro_batch_size_per_gpu=2 \
    custom_reward_function.path=thirdparty/verl/sf_scripts/skill_factory_rewards.py

# run grpo (1 node 2 gpu), countdown_3arg
# NOTE: change trainer.project_name to log into your own wandb project
unset ROCR_VISIBLE_DEVICES && CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.logger=['console','wandb'] \
    trainer.project_name=jackrl \
    trainer.experiment_name=test \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=10 \
    trainer.total_epochs=100 \
    data.train_files=/scratch/yl11330/skill-factory/data/countdown_3arg/train.parquet \
    data.val_files=/scratch/yl11330/skill-factory/data/countdown_3arg/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=4096 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.model.path=TAUR-dev/SIE-countdown3arg_sft1ep_1e6lr_mix_ss_pse_vote_ansrev-sft \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    custom_reward_function.path=thirdparty/verl/sf_scripts/skill_factory_rewards.py