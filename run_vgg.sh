python train.py --use_mask False --feat_position 'mutual' --temperature 0.1 --gpu_id 0 --choose_network 'resnet' --name_exp 'wo_mask_train_only_agg' \
--alpha_1 0.05 --alpha_2 2.5 
#python train.py --use_mask False --feat_position 'mutual' --temperature 0.05 --gpu_id 0 --choose_network 'resnet' --name_exp 'resnet_mutual'
#python train.py --use_mask False --feat_position 'feat' --temperature 0.05 --gpu_id 0 --choose_network 'vgg' --pretrained 'Semantic_GLUNet_DPED_CityScape_ADE.pth'
#python train.py --use_mask False --use_adap True --feat_position 'mutual' --temperature 0.05 --gpu_id 0 --choose_network 'vgg' --pretrained 'PDCNet_megadepth.pth.tar'
#python train.py --use_mask False --use_adap True --feat_position 'mutual' --temperature 0.05 --gpu_id 0 --choose_network 'vgg' --pretrained 'GLUNet_GOCor_star_megadepth.pth'

