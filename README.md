# dataset_embeddings_2.0

This is an updated version of [this](https://github.com/BYUDML/dataset_embeddings) repository. It includes several new updates as well as minimal changes to the existing code. It also confirms and improves results. Below follows a more detailed explanation of files and experiments to accompany the paper (currently only located on Overleaf). 

## Brief Overview of Changes
added:
 - Denoise-Transformer-AutoEncoder
 - IGTD
 - Pytorch-scarf

updated:

 - Tabnet
	 - While this was included in the previous repository no experimentation or implementation of the package had been done. To preserve the original nature of the project I created a new folder named tabnet_2.0 and included my experimentation there. 

Additionally, some minor updates and some changes to the organization of the final meta-model experiments are also included. The most major change was including kfold cross-validation for the accuracies. Previously the accuracies generated were in the 20-30% range. After they were 50-70% range.

## Brief Description of What Worked and Didn't and Why
Most of the things that did work are detailed in the paper. But this section is to also document what didn't work(in alphabetical order).

**Data2Vec**
This was largely implemented by Joshua and needed no further updates. I struggled to fully duplicate his exact results (for some reason it would fail on ~every 3rd dataset) but for the embeddings that were generated, I did confirm the results. 

Because the results were not very high I looked into retraining the model on our data and then generating the embeddings. Turns out the model is intentionally built to make this difficult since it can lead to overfitting. Additionally, I experimented with HPO. The paper did give some examples of this working, but it was not implemented fully into the model yet. I decided to hold off on building my own HPO system until after implementing other embedding models. 

**DTA**
This embedding was suggested and implemented entirely by yours truly. I found it when looking at [this](https://sebastianraschka.com/blog/2022/deep-learning-for-tabular-data.html#denoising-autoencoders-daes-for-tabular-data-2021-04) list. Unlike the other embeddings in this project, this does not have an accompanying paper. Additionally, I am unsure if the GitHub repo used is used in the Kaggle competitions. It certainly is not the full version. 

I started by duplicating the results. I immediately found I couldn't run it on the machine I was working on because it had no GPU. Additionally, the code documentation explained it would take 22 hours to train on a single GPU. Even after changing machines and decreasing the epochs to 1/4 of the original it still took 6 hours. Lastly,  the embeddings it generated were huge. 

Because of the computation time and size of the embeddings I didn't get the chance to create a meta dataset. However, this is a very interesting option and should be looked into more. 

**IGTD**
IGTD is a novel approach to handling tabular data by converting it to an image. I was able to create a meta dataset off these embeddings and generate accuracies.  These accuracies were very mediocre, with the best (unsurprisingly) being a neural network at 63%.  

Additionally, IGTD had two major drawbacks from one feature: the size of the embeddings. Each image depended on the size of the dataset. Because of the variability in the size of the embeddings, the final dataset had a considerable number of nan values. This likely heavily affected the performance of many of the algorithms. Additionally, since the image was the embedding the embeddings were quite large. So large I was unable to upload the exact file to the repository. However, all the code to recreate it is included. 

Since the Neural network performed so well, I experimented with several different HPOs to see if I could produce better results. I created a basic grid search and used H20.ai. The grid search was very preliminary and unfortunately didn't generate any better results. H20.ai was unable to finish most likely because of the above issues. 

**NS**
From my perspective, it appeared the Neural Statistician was already implemented in one of the PYMFE versions and therefore I didn't implement it to avoid duplicating work.

**PYMFE**
This was already implemented by Joshua and needs no further updates. This is the baseline, the non-deep learning approach to embeddings, and is what all embedding accuracies are compared against. 

**Tabnet**
This is a very interesting embedding method. While it is built for tabular data, it only generates embeddings for categorical data. Because 232 of the ~466 datasets had at least one categorical feature (not including the target) this was not a viable embedding for this project. It could be interesting to combine this embedding with a numerical only embedding. 

Footnotes: 
 - I had major problems with package requirements. Turns out the download already in the project was several versions behind :( major oops.
 - I answered a question on [SO](https://stackoverflow.com/questions/74683573/how-to-extract-tabnet-embeddings) about getting the embeddings from Tabnet.

**Task2Vec**
This was previously included in the project by Joshua but hadn't been implemented yet. It is built for image data (and NOT tabular data). Because of this, I did not implement it.

**PyTorch-SCARF**
I was able to generate SCARF embeddings on my CPU machine but it failed to run on a GPU machine. After spending some significant time debugging and researching these errors the only solution I found was to switch back to the CPU machine. I even asked about this on [SO](https://stackoverflow.com/questions/76434294/pytorch-scarf-package-runtimeerror-expected-all-tensors-to-be-on-the-same-devic). 

Like IGTD, the embeddings are based on the size of the dataset making it difficult to train. Additionally, SCARF was not built for categorical data and needed to be converted to a numerical format. While, this was possibly an easy solution, after running into several frustrations with the formatting of the arff files and the data cleaning, I decided to end experiments with SCARF for the time being. 
 
## Final Footnotes on Future Work and Conclusion
One of the areas of this project I struggled with the most was getting caught up with debugging issues/errors. Additionally, finding a new embedding and then getting my hopes dashed when it only worked for categorical features or it created an embedding based on the size of the dataset, etc. was also frustrating. 


I presented this project at the BYU Student Research Conference. A copy of my slides can be found [here](https://www.canva.com/design/DAFby8lhl-Q/cN4hS5AqHW1uGQp7jpfxdw/view?utm_content=DAFby8lhl-Q&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink). 

