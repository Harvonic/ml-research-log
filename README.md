# ML Research Log

A collection of independent experiments exploring ideas in machine learning, with a focus on neural network architectures, training efficiency, and language modelling.

This repository serves as a public research log where I document questions I'm interested in, test hypotheses through small-scale experiments, and share the results. The goal isn't necessarily to achieve state-of-the-art performance, but to better understand how neural networks work and explore alternative approaches to existing methods.

Experiments include both successful and unsuccessful results. Each is documented with its motivation, implementation, methodology, and findings.

## Repository Structure

Each experiment is self-contained and generally follows the structure below:

```text
experiments/
└── 001-experiment-name/
    ├── README.md
    ├── MOTIVATION.md
    ├── report.pdf
    ├── src/
    ├── results/
    ├── config.yaml
    └── requirements.txt
```

- **README.md:** Overview, methodology, key findings, and reproduction instructions.
- **MOTIVATION.md:** Original reasoning, research questions, and hypotheses.
- **report.pdf:** Formal technical report detailing the experiment and its results.
- **src/:** Source code for models, training, and evaluation.
- **results/:** Experimental measurements and visualizations.
- **config.yaml:** Experimental configurations and hyperparameters.
- **requirements.txt:** Dependencies required to reproduce the experiment.

Some experiments may differ slightly depending on their scope.

## Methodology

I aim to follow a consistent process across experiments:

1. Identify a question or hypothesis.
2. Design an experiment with appropriate baselines and evaluation metrics.
3. Implement and run the experiment.
4. Analyze the results, including unexpected outcomes and limitations.
5. Document the findings and identify potential directions for further investigation.

Most experiments are relatively small in scale and intended as preliminary investigations rather than definitive evaluations.

## AI Assistance

AI tools are used throughout parts of the research process, including brainstorming, implementation, experiment design, analysis, and technical writing.

The reports aim to distinguish experimental observations from hypotheses and interpretations. Results should be treated in the context of the methodology and limitations described in each experiment.

## Notes

This is an ongoing personal research project. Experiments may be incomplete, exploratory, or later revised as new findings become available.

The inclusion of an idea or result does not imply that it is novel or that it has undergone peer review.
