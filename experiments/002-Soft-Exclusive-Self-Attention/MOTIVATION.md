Some dude at Apple released a paper earlier this year called Exclusive Self Attention (XSA) (https://arxiv.org/abs/2603.09078), where he basically modified 2 lines of code and that caused improvements in model performance across multiple experiments.

It works by removing the projection of the value vector from its corresponding output vector after attention. The idea is that this prevents the model from relying too heavily on a token's own information, instead encouraging it to extract more information from the surrounding tokens. This should lead to better contextual understanding, rather than simply preserving information the model already has.

The basic formula is z = y - proj_v(y), where y is regular attention.

I wanted to test whether theres an optimal "strength" of this reduction. ie. z = y - lambda * proj_v(y), where 0 <= lambda <= 1, or potentially even checking lambda > 1. 

Apparently, someone already experimented with a learnable version of this, and found that regular XSA (lambda=1) performed better. Although this was only tested on a small model, so it doesn't necessarily mean lambda=1 is optimal.

I still wanna try it anyways, especially with lambda values greater than 1. Interestingly, lambda=2 would actually reverse the component being removed, rather than just removing it more strongly.

I don't really have a hypothesis for which value will perform best, so I'm just gonna predict lambda=2 because it would be cool if it worked.


Related Stuff:

-https://gitlab.uni-osnabrueck.de/mnatur/deep-learning-challenge-sose-2026/-/blob/master/PAPER.md

-https://github.com/turancannb02/xsa-marimo-notebook