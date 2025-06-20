## Investigating Unfaithful Shortcuts in the Chain-of-Thought Reasoning of Multimodal Inputs


### Report prepared for the [Whitebox Research](https://www.whiteboxresearch.org/) Interpretability Fellowship 2025 - Project Phase
Recent studies show that Chain-of-Thought (CoT) reasoning by Large Language Models isn't always faithful, the reasoning steps don't necessarily reflect actual internal processes. This poses challenges for AI safety and interpretability. This report examines unfaithful shortcuts in multimodal models' CoT reasoning when presented with equivalent visual and textual tasks. We evaluate Gemini 2.0 Flash Experimental and Claude 3.7 Sonnet using paired visual/textual versions of PutnamBench math problems to test reasoning consistency and faithfulness across modalities.

Please see the [report](https://github.com/whitebox-research/c2-proving-ground-martinez-cot/blob/main/report.pdf) for more details.

Prepared by Ishita Pal, Aditya Thomas & Angel Martinez. Mentored by Angel Martinez.


### Results

We find that of the models tested, Google Gemini 2.0 Flash Experimental (with thinking enabled) and Anthropic Claude 3.7 Sonnet (with extended thinking), there was no difference in the accuracy of the results for images versus the text inputs (see section 3 Results oand Discussion of our report) for our dataset.

![Faithful Metric - Claude](/plots/claude_unfaithfulness_plot.png?raw=true "Faithful Metric - Claude") 

![Faithful Metric - Gemini](/plots/gemini_unfaithfulness_plot.png?raw=true "Faithful Metric - Gemini") 

The distribution of our “Faithful Metric” across the two modes for the two models is similar. The scores concentrating around 2 or 3 for both models suggest some degree of reliance on partial shortcuts. However, complete failures or perfect reasoning were also shown to be uncommon. Severely unfaithful steps are rare and so are highly faithful ones. More large-scale analysis is needed to validate these trends in various tasks and domains.

### Usage

For how to install and run the scripts, please see Append D of the [report](https://github.com/whitebox-research/c2-proving-ground-martinez-cot/blob/main/report.pd).


### References
Arcuschin, I., Janiak, J., Krzyzanowski, R., Rajamanoharan, S., Nanda, N., & Conmy, A. (2025). Chain-of-thought reasoning in the wild is not always faithful. arXiv preprint arXiv:2503.0867

Tsoukalas, G., Lee, J., Jennings, J., Xin, J., Ding, M., Jennings, M., ... & Chaudhuri, S. (2024). Putnambench: Evaluating neural theorem-provers on the putnam mathematical competition. arXiv preprint arXiv:2407.11214.
