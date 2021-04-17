epsilon=("5.0" "1.0" "0.5" "0.1")
for ((i=0;i<${#epsilon[@]};++i)); do
    epsilon_val=${epsilon[i]}
    epsilon_str="--epsilon=${epsilon_val}"

    log_val="Optimized/MNIST_Optimized_epsilon=${epsilon_val}.log"

    log_21_val="train_${epsilon_val}.log"

    comm="nohup python -u main_clip_new.py $epsilon_str > $log_val"
    echo $comm 2>&1 | tee $log_21_val
    $comm 2>&1 | tee -a $log_21_val

done