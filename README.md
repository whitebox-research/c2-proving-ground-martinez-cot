## Investigating Unfaithful Shortcuts in the Chain-of-Thought Reasoning for Multimodal Inputs


### Report prepared for the [Whitebox Research](https://www.whiteboxresearch.org/) AI Interpretability Fellowship 2025 - Project Phase
Chain-of-Thought (CoT) reasoning by Large Language Models is not always faithful, i.e., the reasoning steps don't necessarily reflect actual internal processes. Simultaneously, there has been a rapid uptake of multimodal models that
integrate both visual and textual information. These systems pose new challenges and opportunities for studying CoT reasoning across modalities.

This report examines unfaithful shortcuts in multimodal models' CoT reasoning when presented with semantically equivalent visual and textual tasks. We evaluate Gemini 2.0 Flash Experimental and Claude 3.7 Sonnet using paired, semantically equivalent visual and textual versions of PutnamBench math problems to test the faithfulness of reasoning and performance across modalities.

Please see the [report](https://github.com/whitebox-research/c2-proving-ground-martinez-cot/blob/main/report.pdf) for more details.

Prepared by Ishita Pal, Aditya Thomas & Angel Martinez. Mentored by Angel Martinez.


### Results

We find that for the models tested, Google Gemini 2.0 Flash Experimental (with thinking enabled) and Anthropic Claude 3.7 Sonnet (with extended thinking), there was no difference in the accuracy of the results for the text and semantically similar image inputs (see section 3 Results and Discussion of our report).

<table>
  <thead>
    <tr>
      <th></th>
      <th colspan="2">Claude 3.7 Sonnet (Thinking Mode)</th>
      <th colspan="2">Gemini 2.0 Flash Experimental (Thinking Mode)</th>
    </tr>
    <tr>
      <th>Metric</th><th>Texts</th><th>Images</th><th>Texts</th><th>Images</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Problems analysed</td><td>27</td><td>27</td><td>181</td><td>181</td></tr>
    <tr><td>Correct responses</td><td>24</td><td>26</td><td>114</td><td>110</td></tr>
  </tbody>
</table>


The distribution of our “Faithful Metric” across the two modes for the two models is similar. The scores concentrating around 2 or 3 for both models suggest some degree of reliance on partial shortcuts. However, complete failures or perfect reasoning were also shown to be uncommon. Severely unfaithful steps are rare and so are highly faithful ones. More large-scale analysis is needed to validate these trends in various tasks and domains.

![Faithful Metric - Claude](/plots/claude_unfaithfulness_plot.png?raw=true "Faithful Metric - Claude") 

![Faithful Metric - Gemini](/plots/gemini_unfaithfulness_plot.png?raw=true "Faithful Metric - Gemini") 


### Usage

For how to install and run the scripts, please see Appendix D of the [report](https://github.com/whitebox-research/c2-proving-ground-martinez-cot/blob/main/report.pd).


### Acknowledgements
We gratefully acknowledge the support of the AI Interpretability Fellowship from WhiteBox Research, which provided both training in interpretability and evaluation methodologies, as well as financial support for this project. We also thank Angel Martinez for conceptualizing the project and providing guidance throughout all stages of its development.


### References (partial)
Arcuschin, I., Janiak, J., Krzyzanowski, R., Rajamanoharan, S., Nanda, N., & Conmy, A. (2025). Chain-of-thought reasoning in the wild is not always faithful. arXiv preprint arXiv:2503.0867

Tsoukalas, G., Lee, J., Jennings, J., Xin, J., Ding, M., Jennings, M., ... & Chaudhuri, S. (2024). Putnambench: Evaluating neural theorem-provers on the putnam mathematical competition. arXiv preprint arXiv:2407.11214.
