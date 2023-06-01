# TMPCC

Y. Cai, et al., "Transformer-based Contrastive Prototypical Clustering for Multimodal Remote Sensing Data", under consideration at Inf. Sci.

## Requirements ##

    pip install torch==1.10.x torchvision==0.11.x spectral PyYAML scikit-learn

## Run ##
    python train.py


Configurations  are defined in config.yaml.

    # general
    seed: 42
    workers: 2
    
    
    dataset: "Trento"
    dataset_root: 'HSI-Lidar-Trento\\'
    model_path:  "save/Trento"
    
    
    # train options
    batch_size: 256
    image_size: 7
    joint_train_epoch: 20
    dim_emebeding: 512
    lr_scale: 10
    
    is_labeled_pixel: False
    is_pretrain: False
    
    # loss options
    learning_rate: 0.0005 #0.00002
    weight_decay: 0.0005
    contrastive_param: 0.5 # tau
    weight_clu_loss: 1  # lambda
    regularizer_coef: 0.00001  # gamma
    
    


