# relearning_attack_grpo
Evaluating large language model (LLM) unlearning requires determining whether
target information remains accessible after its removal. However, unlearning is
commonly evaluated and attacked through single responses, even though target
information may remain exposed under repeated sampling. This probabilistic
exposure creates an additional attack surface beyond single-response relearning
attacks. We introduce the first probabilistic unlearning attack that directly op-
timizes target-fact disclosure over sampled responses. We derive finite-sample
confidence bounds, characterize its idealized dynamics, and establish conditions
for greater disclosure than single-reference supervised fine-tuning (SFT). Across
Harry Potter QA, TOFU, and WMDP, we show that single-response metrics can
miss target exposure, GRPO recovers unlearned information more effectively than
state-of-the-art SFT, and it substantially compromises RULE-NPO, a recent method
designed to resist both relearning attacks and probabilistic leakage. Our results
reveal probabilistic outputs as an important attack surface for LLM unlearning.
