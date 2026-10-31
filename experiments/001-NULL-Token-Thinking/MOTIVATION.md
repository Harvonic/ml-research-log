I was thinking that allowing "NULL" tokens could let an LLM perform latent reasoning without requiring visible chain of thought. The idea is that generating these tokens gives the model additional computational steps before producing an answer.

For example, one must imagine the following equation to be extremely difficult:
1 + 100

Under the assumption, naturally, an LLM might struggle, but visible chain of thought might act as a scratchpad that keeps getting appended as input, helping the model reason better, but what if that is unnessesary, what if just having it output "NULL" tokens could still improve reasoning as the model has more opportunities to compute using its hidden representations, effectively using them as a scratchpad.

For example (I'm just making these outputs up to illustrate the idea, don't take them too seriously):

Normal CoT:
<User> 1 + 100 = <User>
<Model> 
    <Thinking> 0 + 1 = 1, but wait we have 100, then 0 + 1 + 100, if we replace the first 0 in the 100, then 0 + 101 = 101 <Thinking> 
    1 + 100 = 101
<Model>

No CoT:
<User> 1 + 100 = <User>
<Model> 1 + 100 = 1100 <Model>

Null CoT:
<User> 1 + 100 = <User>
<Model> 1 + 100 = NULL NULL NULL NULL NULL 101 <Model>

Another interesting consequence if NULL thinking works is that regular CoT might not actually reveal what a model is thinking. For example if a model can reason in its hidden layers, the visible CoT might not be a representation of what it is actually thinking. Of course, NULL reasoning working wouldn't prove that CoT is unfaithful, but I think it raises an interesting question.

Ultimately, I want to see whether a model can learn to use these otherwise meaningless tokens as a computational scratchpad, and whether this actually improves its reasoning compared to a model without them.

Anyways thats just a theory, an ML theorrrrry