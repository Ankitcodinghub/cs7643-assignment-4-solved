# cs7643-assignment-4-solved
**TO GET THIS SOLUTION VISIT:** [CS7643 Assignment 4 Solved](https://www.ankitcodinghub.com/product/cs7643-deep-learning-assignment-4-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;126113&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;5&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (5 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CS7643  Assignment 4 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (5 votes)    </div>
    </div>
• It is your responsibility to make sure that all code and other deliverables are in the correct format and that your submission compiles and runs. We will not manually check your code (this is not feasible given the class size). Thus, non-runnable

code in our test environment will directly lead to a score of 0. Also, your entire programming parts will NOT be graded and given 0 score if your code prints out anything that is not asked in each question.

Theory Problem Set

1. Given a 4 length sequence RNN, compute and show how you derived it. In addition, please provide its computation graph. The following equations might be helpful:

at = Uxt + Wht−1 + b

ht = tanh(at) ot = V ht + c

yˆt = Softmax(ot)

Lt = CE(yˆt,yt)

0 f(x) = tanh(x) −→ f (x) = 1− tanh2(x)

Hint: You can combine softmax and CE into a single derivative as follows:

Paper Review

In this section, you must choose one of the papers below and complete the following:

1. provide a short review of the paper,

2. answer paper-specific questions,

Guidelines: Please restrict your reviews to no more than 350 words and answers to questions to no more than 350 words per question. The review part (1) should include answers to the following:

3. What is the main contribution of this paper? In other words, briefly summarize its key insights. What are some strengths and weaknesses of this paper?

Paper Choice 1:

Up until recently, convolutional neural networks (CNN) have been the state of the art for computer vision (CV) tasks. With the recent introduction of vision transformers (ViT), there has been research into using this new architecture for CV. There have been instances where ViTs have either achieved similar or even superior accuracy to CNNs. The biggest question is, how do they see? Is it similar to CNNs? Is it different? This paper attempts to answer these questions.

The paper can be viewed here.

Questions for this paper:

• Compare and contrast the learned features of ViTs and CNNs? For differences between the two, please provide explanations in terms of network architecture and training.

• What is meant by spatial localization? And why might we consider the use of ViTs better for object detection?

Paper Choice 2:

The second paper focuses on the generalization ability of pre-trained language models, specifically zero-shot learning (i.e., performing a new task with no labels at all). It turns out these models (like GPT-3) are surprisingly effective zero-shot learners. The paper can be viewed here.

Questions for this paper:

• How might we approach the technical limitations (e.g., changes in architecture, other context/data, optimization, etc.) mentioned in sections 5/6?

• What are the social implications of deploying these models for various uses (e.g., to generate image captions, answer questions as chatbots, etc.)?

1 Code: RNN, LSTM, Seq2Seq, and Transformer models

In this assignment, we will work with Python 3. If you do not have a python distribution installed yet, we recommend installing Anaconda (or mini-conda) with Python 3 (3.8.10 recommended).We have provided you with a conda environment YAML file to assist you in setting up the code environment. Alternatively, there is a requirements.txt file if you decide to use pip to install the required packages. You should make sure you have the following packages installed

$ pip install torchtext==0.14.1

$ pip install torch==1.13.1

$ pip install spacy==3.4.4

$ pip install tqdm

$ pip install numpy

Additionally, you will need the Spacy tokenizers in English and German language, which can be downloaded as such:

$python -m spacy download en_core_web_sm

$python -m spacy download de_core_news_sm

In your assignment you will see the notebook Machine_Translation.ipynb which contains test cases and shows the training progress. You can follow that notebook for instructions as well.

2 RNNs and LSTMs

In models/naive you will see files necessary to complete this section. In both of these files you will complete the initialization and forward pass.

2.1 RNN Unit

You will be using PyTorch Linear layers and activations to implement a vanilla RNN unit. Please refer to the following structure and complete the code in RNN.py:

2.2 LSTM

You will be using PyTorch nn.Parameter and activations to implement an LSTM unit. You can simply translate the following equations using nn.Parameter and PyTorch activation functions to build an LSTM from scratch:

Here’s a great visualization of the above equation from Colah’s blog to help you understand LSTM unit.

If you want to see nn.Parameter in example, check out this tutorial from PyTorch.

3 Seq2Seq Implementation

In models/seq2seq you will see the files needed to complete this section. In these files you will complete the initialization and forward pass in __init__ and forward function.

Encoder.pyDecoder.pySeq2Seq.py

3.1 Seq2Seq with attention

We will be implementing a simple form of attention to evaluate how it impacts the performance of our model. In particular, we will be implementing cosine similarity as the attention mechanism in decoder.py per this diagram. Please pay attention to comments in TODO sections of the code for more detail.

To learn more about attention and how it is used in Neural Machine Translation, we recommend (but not required) that you read “Neural machine translation by jointly learning to align and translate (Bahdanau et al)” and “Effective Approaches to Attentionbased Neural Machine Translation (Luong et al)”.

3.2 Training and Hyperparameter Tuning

Train seq2seq on the dataset with the default hyperparameters. Then perform hyperparameter tuning and include the improved results in a report explaining what you have tried. Do NOT just increase the number of epochs or change the model type (RNN to LSTM) as this is too trivial.

4 Transformers

We will be implementing a one-layer Transformer encoder which, similar to an RNN, can encode a sequence of inputs and produce a final output of possibility of tokens in target language. The architecture can be seen below.

You can refer to the original paper. In models you will see the file Transformer.py. You will implement the functions in the TransformerTranslator class.

4.1 Embeddings

Your first task is to implement the embedding lookup, including the addition of positional encodings. Complete the code section for Deliverable 1, which will include part of __init__ and embed.

4.2 Multi-head Self-Attention

Attention can be computed in matrix-form using the following formula:

We want to have multiple self-attention operations, computed in parallel. Each of these is called a head. We concatenate the heads and multiply them with the matrix attention_head_projection to produce the output of this layer.

After every multi-head self-attention and feedforward layer, there is a residual connection + layer normalization. Make sure to implement this, using the following formula:

Implement the function multi_head_attention for Deliverable 2. We have already initialized all of the layers you will need in the constructor.

4.3 Element-Wise Feedforward Layer

Complete code for Deliverable 3 in feedforward_layer: the element-wise feed-forward layer consisting of two linear transformers with a ReLU layer in between.

4.4 Final Layer

Complete code for Deliverable 4 in final_layer., to produce probability scores for all tokens in target language.

4.5 Forward Pass

Put it all together by completing the method forward, where you combine all of the methods you have developed in the right order to perform a full forward pass.

4.6 Training and Hyperparameter Tuning

Train the transformer architecture on the dataset with the default hyperparameters – you should get a perplexity better than that for seq2seq. Then perform hyperparameter tuning and include the improved results in a report explaining what you have tried. Do NOT just increase the number of epochs as this is too trivial.

5 Deliverables

You will need to submit the notebook as well as your code in the models folder. Run the script collect_submission.py to generate a zip folder with all the required code files. Upload the resulting zip folder to Gradescope.

You will need to follow the guidance and fill in the report template.Your report should include the performance metrics of the Seq2Seq model and Transformer architecture before and after hyper-parameter tuning with explanations of what you did and why. You also need to include the answers to the theory question and the section on paper review in your report.When submitting to Gradescope, make sure you select

ALL corresponding slides for each question. Failing to do so will result in point deductions.
