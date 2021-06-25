noise=("AdaClip") #  "AdaClip" "Optimized_Normal" "Optimized_Special" "Gaussian"
epsilon=("0.4") #"0.1" "0.2" "0.3" "0.4" "0.5" "1.0" "2.0" "5.0" "10.0"
for ((i=0;i<${#epsilon[@]};++i)); do
    epsilon_val=${epsilon[i]}
    epsilon_str="--epsilon=${epsilon_val}"

    for ((j=0;j<${#noise[@]};++j)); do
    	noise_val=${noise[j]}
    	noise_str="--noise_type=${noise_val}"

	    log_val="Optimized/MNIST_${noise_val}_epsilon=${epsilon_val}.log"
	    log_str="--log_name=${log_val}"

	    log_21_val="train_${epsilon_val}.log"

	    comm="python -u main.py $epsilon_str $noise_str $log_str"
	    echo $comm 2>&1 | tee $log_21_val
	    $comm 2>&1 | tee -a $log_21_val

	done
done
