---
title: 🗓️ Generating Realistic Dialogues in OR using Large Language Models 
date: 2025-02-07 15:00:00 +/-0800
categories: [Publication, Programming]
---

Operating Rooms (ORs) are most comlicated and sensitive environments in any medical center. Providing best possible healthcare service requires close team collabration of medical personel, including physicians and assistants.

In this post, we will investigate the collabration work during port-catheter placement operation in the context of surgical workflow analysis (SWA). The port-catheter placement operation is performed by a radiologist and a medical assistant. And patient is not under full anesthesia during the procedure, so they can talk. Thus, dialogues in OR occur among these three person. A detailed explanation of the entire procedure is given in our previous post [\link] Port-Cather Placement in Surgical Data Science [\link].

We can utiliize information extracted from these conversations to understand performed actions in the OR. We can even train or finetune a BERT-like language model specifically for this task and solve many medical language problems successfully. This will open ways for many new applications in the OR.

However, collecting these conversational data in OR does not come cheap. Setting up actual recording setup and undertaking recording work require huge amount of resources and long time. We recorded ~40 operations in six months, which has approximately 40k tokens. Compared to dataset sizes we know in NLP community, it is a small amount.

We can use the power of generative language models to overcome this problem. To be specific, we can use these models to mimic dialogues in our real dataset. Language models will adapt to randomly selected personalities of physicians, medical assistants, and patients, then, generate dialogues as real personalities are having conversations in the OR.

![Desktop View](/assets/img/syn1.png){: w="500" h="300" }

In this work, we used Gemma2-27b as the LLM as it is publicly available in Huggingface and offers good size-performance trade-off in German. Entire source code is availabe on [GitHub](https://github.com/kubicndmr/TextualSPR).

Our aim to achieve augmentation of the real dataset. But due to unavoidable sim-to-real gap, this may not be 100% possible. Main target we want to achieve at this point is to increase similarity of real and synthetic datasets as much as possible. We consider the points below for the generation of surgical dialogues:

##### 1. Class Distribution: 

To ensure the alignment of generated synthetic conversations with the distribution of real data, we constrained LLMs to produce a controlled number of utterances for each surgical phase.

##### 2. Personae:

To enhance natural variation within the syn- thetic dataset and reduce overfitting during model training, we created 100 unique personas for the radiologist, the assistant, and the patient. For each simulated operation, a randomly selected persona was assigned to the LLM. The language model was instructed to consistently adopt the assigned persona’s communication and operational style throughout the dialogue.

##### 3. Domain Knowledge:

While the primary objective of our study is the recognition of surgical phases, relying solely on phase-level information is insufficient for generating medically accurate dialogues. To address this limitation, we also incorporated detailed surgical step information into the generation prompts. This additional information includes the specific actions required to complete each surgical step and explanations of the objectives behind these actions.

##### 4. Daily Conversations:

During surgical procedures, radiologists often engage in conversations with patients about daily, non-medical topics to reduce stress, provide encourage- ment, and serve as a distraction [34]. To replicate this behavior in our simulations, we instructed the LLMs to include out-of- topic (OOT) conversations during data generation. To prevent language models from repeatedly mentioning the same generic daily topics across simulations, we curated a set of 100 diverse suitable topics for discussion during operations.

##### 5. Interaction of LLMs:

Our primary focus is on radiologists’ conversations, which are most relevant to the task of the SPR. However, simulating only sections of radiologist’s con- versations without interaction is not intuitive. The assistant and patient characters were included in the OR simulation to generate realistic and natural dialogues.

##### 6. Complications:

In the OR, unexpected outcomes can arise due to numerous variables. These irregularities may result from patient complications, issues with medical instruments, or OR utilities. To simulate these real-world irregularities, we created a set of 100 examples representing potential problems in OR.

##### 7. Language:

We implemented several measures to generate sentences that closely resemble real, natural-sounding dialogues. Please see the prompt for details.

##### 8. Preserving Temporal Connections:

We instructed language model to follow four-step approach to generate consistent dialogues. 

- The model examines examples of actual conversations; 
- It reviews previously generated dialogues; 
- It evaluates conversations already generated and the remaining quota; 
- It proceeds to create sentences after completing this analysis.

So, did this approach work? Let's look at the generated samples and compare with the original data. To do this investigation, we embedded each sentence into a feature space using E5-large text-embedding language model. Then, we plot sentences within each phase in 2D space using t-SNE plots.

![Desktop View](/assets/img/syn2.jpg){: w="600" h="400" }

Seems like they are overlapping. That's good news! However, it is still not end of the story. If we look at the embeddings in operations level (it means we create an embedding for each dialogue set corresponding a surgical operation by averaging all embeddings in an operation), we see a clear discreapancy. 

![Desktop View](/assets/img/syn3.png){: w="300" h="150" }

This discrepancy was expected. We couldn't achieve the augmentation effect we were looking for, but, the generated synthetic dataset is still very valuable. Transfer learning is the suitable method in this case. We can use this dataset to pretrain our classification model. There is still research going on with domain adaptation methods but it is difficult to apply this technique on text embeddings while keeping their semantic and temporal information intact. Domain randomization is another path and can be applied in this setting. Imagine that there are not just two distributions in the last figure but many. In this way, classifcation algorithm will not overfit to one distribution but generalize better. This can be achieved by applying our technique into different languages, operation types etc. and mergind all datasets. We leave this task for future work.

Let's see how transfer learning method worked in our dataset.

![Desktop View](/assets/img/syn4.png){: w="400" h="175"}

We achieved 5 points increase in all metrics when we pretrained our classifcation model with 400 synthetic operations! This is a noticable gain in eight-class classifcation problem. We hope that our method can help many NLP-based OR applications in the future.