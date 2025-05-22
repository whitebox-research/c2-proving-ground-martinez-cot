# c2-proving-ground-martinez-cot



`python -m scripts.putnam.putnamlike0_images_save_rollouts chainscope\data\putnam2\image_pipeline.yaml --model_id "anthropic/claude-3.7-sonnet_20k" --max_retries=3 --verbose --max_parallel=1`

`python -m scripts.putnam.putnamlike1_are_rollouts_correct chainscope\data\cot_responses\instr-v0\default_sampling_params\filtered_putnambench\claude-3.7-sonnet_10k_v1_prefix_4.yaml --model_id "anthropic/claude-3.7-sonnet_20k" --max_retries=3 --verbose --prefix=10 --max_parallel=3`

`python -m scripts.putnam.putnamlike2_split_cots chainscope\data\cot_responses\instr-v0\default_sampling_params\filtered_putnambench\claude-3.7-sonnet_10k_v1_prefix_4_just_correct_responses.yaml --model_id "anthropic/claude-3.7-sonnet_20k" --max_retries=3 --verbose --prefix=10 --max_parallel=3 `

`python -m scripts.putnam.putnamlike2p5_critical_steps_eval chainscope\data\cot_responses\instr-v0\default_sampling_params\filtered_putnambench\claude-3.7-sonnet_10k_v1_prefix_4_just_correct_responses_splitted.yaml --model_id "anthropic/claude-3.7-sonnet_20k" --max_retries=3 --verbose --max_parallel=3 `

`python -m scripts.putnam.putnamlike3_main_faithfulness_eval chainscope\data\cot_responses\instr-v0\default_sampling_params\filtered_putnambench\claude-3.7-sonnet_10k_v1_prefix_4_just_correct_responses_splitted.yaml  --critical_steps_yaml chainscope\data\cot_responses\instr-v0\default_sampling_params\filtered_putnambench\claude-3.7-sonnet_10k_v1_prefix_4_just_correct_responses_splitted_anthropic_slash_claude-3_dot_7-sonnet_critical_steps.yaml --model_id "anthropic/claude-3.7-sonnet_20k" --max_retries=3 --verbose --max_parallel=3`