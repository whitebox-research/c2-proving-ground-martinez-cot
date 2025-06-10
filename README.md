## Investigating Unfaithful Shortcuts in the Chain-of-Thought Reasoning of Multimodal Inputs

### Report prepared for the Whitebox Research Interpretability Fellowship 2025 - Project Phase
Recent studies show that Chain-of-Thought (CoT) reasoning by Large Language Models isn't always faithful, the reasoning steps don't necessarily reflect actual internal processes. This poses challenges for AI safety and interpretability. This report examines unfaithful shortcuts in multimodal models' CoT reasoning when presented with equivalent visual and textual tasks. We evaluate Gemini 2.0 Flash Experimental and Claude 3.7 Sonnet using paired visual/textual versions of PutnamBench math problems to test reasoning consistency and faithfulness across modalities.

Please see the [report](https://github.com/whitebox-research/c2-proving-ground-martinez-cot/blob/main/report.pdf) for more details.

Prepared by Ishita Pal, Aditya Thomas & Angel Martinez. Mentored by Angel Martinez.

### Usage

Installation:

`pip install -r requirements.txt`

Data processing pipeline:

```python -m scripts.putnamlike0_images_save_rollouts data\dataset\image_pipeline.yaml --model_id "anthropic/claude-3.7-sonnet_20k" --max_retries=2 --max_parallel=1 --verbose```

```python -m scripts.putnamlike1_are_rollouts_correct data\dataset\cot_responses\claude-3.7-sonnet_20k_v0.yaml --model_id "anthropic/claude-3.7" --max_retries=2 --max_parallel=1 --verbose ```

```python -m scripts.putnamlike2_split_cots data\dataset\cot_responses\claude-3.7-sonnet_20k_v0_just_correct_responses.yaml --model_id "anthropic/claude-3.7-sonnet" --max_retries=2 --max_parallel=1 --verbose```

```python -m scripts.putnamlike2p5_critical_steps_eval data\dataset\cot_responsesclaude-3.7-sonnet_20k_v0_just_correct_responses_splitted.yaml --model_id "anthropic/claude-3.7-sonnet" --max_retries=2 --max_parallel=1 --verbose  ```

```python -m scripts.putnamlike3_main_faithfulness_eval data\dataset\cot_responses\claude-3.7-sonnet_20k_v0_just_correct_responses_splitted.yaml  --critical_steps_yaml data\dataset\cot_responses\claude-3.7-sonnet_20k_v0_just_correct_responses_splitted_anthropic_slash_claude-3_dot_7-sonnet_critical_steps.yaml --model_id "anthropic/claude-3.7-sonnet_20k" --max_retries=2 --max_parallel=1 --verbose ```


### References
Arcuschin, I., Janiak, J., Krzyzanowski, R., Rajamanoharan, S., Nanda, N., & Conmy, A. (2025). Chain-of-thought reasoning in the wild is not always faithful. arXiv preprint arXiv:2503.0867

Tsoukalas, G., Lee, J., Jennings, J., Xin, J., Ding, M., Jennings, M., ... & Chaudhuri, S. (2024). Putnambench: Evaluating neural theorem-provers on the putnam mathematical competition. arXiv preprint arXiv:2407.11214.
