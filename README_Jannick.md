This is is a modification of https://github.com/facebookresearch/mae, the PyTorch implementation of the MAE paper.

Modifications include several bug fixes and modernizations for newer module versions, without which the code was not executable.

Further, I've implemented three different masking methods based on Segment-Anything masks:
- segments: multiplies inputs with the mask
- four_channels: multiplies inputs with the mask, and adds the mask itself as a fourth input channel. Expand the input dimension of the ViT model with a fourth channel, which is set to be always 1 after pre-training
- preencoder: add mask as fourth input channel like in four_channels, but compress this down to a 3-channel input volume of otherwise same size. This is done through a "pre-encoder" using a convolution with kernel size equaling the token patch size. The pre-encoder is thrown away after pretraining

### Environment setup
I would recommend Pipenv for the environment, as I tried with Conda but Conda's PyTorch-GPU package was broken at that time. I've included the Pipfile and Pipfile.lock to set this up, but also dumped the packages I used into the requirements.txt.
I ran everything on Python 3.10.

### Running Pretraining
An example of how to run the pretraining on a single GPU can be seen in `run_pretrain.sh`.
The most important parts to adapt here are
- the `--data_path` to the Imagenet-1K dataset. This should be the directory containing the `train`, `val` and `masks` folders, which in turn consist of the training and validation sets, and the Segment-Anything masks for the training set respectively. You can find the dataset on Panther in `/mnt/data0/jannick/img1k`
- the `--mask_method` you want to test, as one of `segments`, `four_channels`, `preencoder`, `patches`, where the latter is the original paper's implementation
- the `--coverage_ratio`, in percent, that we aim for when selecting a mask for an example. The mask with the coverage ratio closest to the given value will be picked
- possibly the `--resume` path when continuing pretraining from a pretraining checkpoint

Most other flags simply correspond to the values recommended by the authors for pretraining in PRETRAIN.md

A checkpoint is automatically created in the job directory every 20 epochs and after the last epoch.

For the first pretraining test, I would use the given settings: the preencoder mask method and a coverage ratio of 15%. The latter is an arbitrary choice that feels reasonable to me. The former seems like the best bet for now, as `segments` doesn't include the mask information, and `four_channels` takes very long to finetune (still looking into this)

### Running Finetuning
An example can be seen in `run_finetune.sh`.

The `--data_path` and `--mask_method` have to be adapted as for pretraining. You need to declare the `mask_method` used for the pretraining here, so that the model checkpoint can be loaded and adapted properly.
Further, the path to the pretrained model to be finetuned should be passed via `--finetune`. The other flags correspond to the values recommended by the authors for pretraining in FINETUNE.md