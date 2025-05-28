# Lifelong Test-Time Adaptation
data=/path/to/imagenet
data_corruption=/path/to/imagenet-c
# data=/data/datasets/ImageNet/
# data_corruption=/data/datasets/ImageNet-C

mkdir -p logs

for algorithm in ptta_eta ptta_eata ptta_deyo; do
    log_filename="logs/$(date +"%Y%m%d_%H%M%S")_${algorithm}_lifelong.log"
    exec > "$log_filename"
    python main_lifelong.py --data ${data} --data_corruption ${data_corruption} --exp_type 'continual' --algorithm ${algorithm} \
        --batch_size 64 --level 5 --loss2_weight 3 --queue_size 1000 --learning_rate 0.001 --neighbor 1 --network vit
done

for algorithm in ptta_cpl; do
    log_filename="logs/$(date +"%Y%m%d_%H%M%S")_${algorithm}_lifelong.log"
    exec > "$log_filename"
    python main_lifelong_cpl.py --data ${data} --data_corruption ${data_corruption} --exp_type 'continual' --algorithm ${algorithm} \
        --batch_size 64 --level 5 --loss2_weight 0.5 --queue_size 1000 --learning_rate 0.001 --neighbor 1 --network vit
done


# Continual Test-Time Adaptation, use ETA as an example
for algorithm in ptta_eta; do
    for network in vit resnet50; do
        log_filename="logs/$(date +"%Y%m%d_%H%M%S")_${algorithm}_${network}_continual.log"
        exec > "$log_filename"
        python main.py --data ${data} --data_corruption ${data_corruption} --exp_type 'continual' --algorithm ${algorithm} \
            --batch_size 64 --level 5 --loss2_weight 3 --queue_size 1000 --learning_rate 0.001 --neighbor 1 --network ${network}
    done
done

# Single Test-Time Adaptation, use ETA as an example
for algorithm in ptta_eta; do
    for network in vit resnet50; do
        log_filename="logs/$(date +"%Y%m%d_%H%M%S")_${algorithm}_${network}_single.log"
        exec > "$log_filename"
        python main.py --data ${data} --data_corruption ${data_corruption} --exp_type 'single' --algorithm ${algorithm} \
            --batch_size 64 --level 5 --loss2_weight 3 --queue_size 1000 --learning_rate 0.001 --neighbor 1 --network ${network}
    done
done

# For ImageNet-A
log_filename="logs/$(date +"%Y%m%d_%H%M%S")_${algorithm}_imagenet-a.log"
exec > "$log_filename"
data_corruption=/data/datasets/ImageNet-A/imagenet-a/
python main_imagenetA.py --dataset imagenet-a --data ${data} --data_corruption ${data_corruption} --exp_type 'each_shift_reset' --algorithm ${method} \
    --batch_size 64 --level 5 --loss2_weight 1 --queue_size 1000 --learning_rate 0.005 --neighbor 1 --network vit

# For ImageNet-R
log_filename="logs/$(date +"%Y%m%d_%H%M%S")_${algorithm}_imagenet-r.log"
exec > "$log_filename"
data_corruption=/data/datasets/ImageNet-R/imagenet-r
python main_imagenetR.py --dataset imagenet-r --data ${data} --data_corruption ${data_corruption} --exp_type 'each_shift_reset' --algorithm ${method} \
    --batch_size 64 --level 5 --loss2_weight 1 --queue_size 1000 --learning_rate 0.005 --neighbor 1 --network vit

# For ImageNet-Sketch
log_filename="logs/$(date +"%Y%m%d_%H%M%S")_${algorithm}_imagenet-s.log"
exec > "$log_filename"
data_corruption=/data/datasets/ImageNet-Sketch/sketch
python main_imagenetsketch.py --dataset imagenet-k --data ${data} --data_corruption ${data_corruption} --exp_type 'each_shift_reset' --algorithm ${method} \
    --batch_size 64 --level 5 --loss2_weight 1 --queue_size 1000 --learning_rate 0.005 --neighbor 1 --network vit

# For ImageNet-V2
log_filename="logs/$(date +"%Y%m%d_%H%M%S")_${algorithm}_imagenet-v.log"
exec > "$log_filename"
data_corruption=/data/datasets/ImageNet-V2/imagenetv2-matched-frequency-format-val
python main_imagenetV2.py --dataset imagenet-v --data ${data} --data_corruption ${data_corruption} --exp_type 'each_shift_reset' --algorithm ${method} \
    --batch_size 64 --level 5 --loss2_weight 1 --queue_size 1000 --learning_rate 0.00025 --neighbor 1 --network vit

# For ImageNet-1k
log_filename="logs/$(date +"%Y%m%d_%H%M%S")_${algorithm}_imagenet-1k.log"
exec > "$log_filename"
data_corruption=/data/datasets/ImageNet-C
python main_imagenet1k.py --data ${data} --data_corruption ${data_corruption} --exp_type 'each_shift_reset' --algorithm ${method} \
    --batch_size 64 --level 5 --loss2_weight 1 --queue_size 1000 --learning_rate 0.005 --neighbor 1 --network vit
