<?php

$n_jobs = 0;


function start_de_job($flight_id, $output_target, $input_lags, $hidden_layers, $nn_type, $population_size, $parent_selection, $n_pairs, $recombination, $bias_name, $use_bias, $task_name) {
    global $n_jobs;

    mkdir("/home/travis.desell/flight_data/output_" . $flight_id);

    $command = "#PBS -S /bin/sh
#PBS -l nodes=4:ppn=8,walltime=03:00:00,naccesspolicy=singlejob
#PBS -o /home/travis.desell/flight_data/output_" . $flight_id . "/out_" . $flight_id . "_" . $output_target . "__de_" . $parent_selection . "_" . $n_pairs . "_" . $recombination . "_p" . $population_size . "__" . $nn_type . "_i" . $input_lags . "_h" . $hidden_layers . $bias_name . "__run" . $task_name . "
#PBS -e /home/travis.desell/flight_data/output_" . $flight_id . "/err_" . $flight_id . "_" . $output_target . "__de_" . $parent_selection . "_" . $n_pairs . "_" . $recombination . "_p" . $population_size . "__" . $nn_type . "_i" . $input_lags . "_h" . $hidden_layers . $bias_name . "__run" . $task_name . "
#PBS -N flight_prediction
#PBS -V

cd ~/flight_analysis/build

/opt/mvapich2-x/gnu/bin/mpiexec -machinefile \$PBS_NODEFILE -np 32 ./artificial_neural_network_so --input_filename /home/travis.desell/flight_data/gecco_data/no_excedence/" . $flight_id . " --input_lags " . $input_lags . " --output_timesteps 1 --hidden_layers " . $hidden_layers . " --network_type " . $nn_type . " --search_type de_mpi --output_target $output_target --population_size " . $population_size . " --parent_type " . $parent_selection . " --number_pairs " . $n_pairs . " --seconds_into_future 0 --maximum_iterations 30000 $use_bias --recombination_selection $recombination --quiet";

    echo "\tout_" . $flight_id . "_" . $output_target . "__de_" . $parent_selection . "_" . $n_pairs . "_" . $recombination . "_p" . $population_size . "__" . $nn_type . "_i" . $input_lags . "_h" . $hidden_layers . $bias_name . "__run" . $task_name . "\n";

    echo "$command\n";
    $script_filename = "/home/travis.desell/submission_scripts/de_script_" . $n_jobs . ".pbs";
    file_put_contents($script_filename, $command);
    shell_exec("qsub $script_filename");
}

function start_ps_job($flight_id, $output_target, $input_lags, $hidden_layers, $nn_type, $population_size, $inertia_weight, $local_best_weight, $global_best_weight, $bias_name, $use_bias, $task_name) {
    global $n_jobs;

    mkdir("/home/travis.desell/flight_data/output_" . $flight_id);

    $command = "#PBS -S /bin/sh
#PBS -l nodes=4:ppn=8,walltime=03:00:00,naccesspolicy=singlejob
#PBS -o /home/travis.desell/flight_data/output_" . $flight_id . "/out_" . $flight_id . "_" . $output_target . "__ps_i" . $inertia_weight. "_l" . $local_best_weight. "_g" . $global_best_weight . "_p" . $population_size . "__" . $nn_type . "_i" . $input_lags . "_h" . $hidden_layers . $bias_name . "__run" . $task_name . "
#PBS -e /home/travis.desell/flight_data/output_" . $flight_id . "/err_" . $flight_id . "_" . $output_target . "__ps_i" . $inertia_weight. "_l" . $local_best_weight. "_g" . $global_best_weight . "_p" . $population_size . "__" . $nn_type . "_i" . $input_lags . "_h" . $hidden_layers . $bias_name . "__run" . $task_name . "
#PBS -N flight_prediction
#PBS -V

cd ~/flight_analysis/build

/opt/mvapich2-x/gnu/bin/mpiexec -machinefile \$PBS_NODEFILE -np \$PBS_NP ./artificial_neural_network_so --input_filename /home/travis.desell/flight_data/gecco_data/no_excedence/" . $flight_id . " --input_lags " . $input_lags . " --output_timesteps 1 --hidden_layers " . $hidden_layers . " --network_type " . $nn_type . " --output_target $output_target --search_type ps_mpi --population_size " . $population_size . " --inertia_weight " . $inertia_weight . " --local_best_weight " . $local_best_weight . " --global_best_weight " . $global_best_weight . " --seconds_into_future 0 --maximum_iterations 30000 $use_bias --quiet
";

    echo "\tout_" . $flight_id . "_" . $output_target . "__ps_i" . $inertia_weight . "_l" . $local_best_weight . "_g" . $global_best_weight . "_p" . $population_size . "__" . $nn_type . "_i" . $input_lags . "_h" . $hidden_layers . $bias_name . "__run" . $task_name . "\n";

    echo "$command\n";
    $script_filename = "/home/travis.desell/submission_scripts/ps_script_" . $n_jobs . ".pbs";
    file_put_contents($script_filename, $command);
    shell_exec("qsub $script_filename");
}



//$flight_ids = array(13588, 15438, 17269, 175755, 24335, 32146, 48551, 48806, 60531, 80789, 83392);
$flight_ids = array(15438, 17269, 175755, 24335);
$output_targets = array("airspeed", "altitude", "pitch", "roll");

/** NEURAL NETWORK OPTIONS **/
$neural_networks = array("feed_forward", "jordan", "elman");
//$neural_networks = array("feed_forward");

$input_lag_options = array(1, 2);
$hidden_layer_options = array(0, 1);
//$bias_options = array("", "--use_bias");
$bias_options = array("--use_bias");


/* OPTIONS FOR DIFFERENT SEARCH TYPES: */
//$search_types = array("de", "ps");
$search_types = array("de");

//$population_sizes = array(250, 500, 1000, 1500);
$population_sizes = array(500);

/* OPTIONS FOR DIFFERENTIAL EVOLUTION */
$parent_selection_options = array("best", "random");
//$recombination_options = array("binary", "exponential");
$recombination_options = array("binary");
//$n_pairs_options = array(1, 2, 3);
$n_pairs_options = array(3);

/* OPTIONS FOR PARTICLE SWARM OPTIMIZATION */
$inertia_weight_options = array(0.85, 0.90, 0.95);
$local_best_weight_options = array(1.75, 2.0);
$global_best_weight_options = array(1.75, 2.0);


echo "starting:\n";


foreach ($flight_ids as $flight_id) {
    foreach ($output_targets as $output_target) {
        foreach ($input_lag_options as $input_lags) {
            foreach ($hidden_layer_options as $hidden_layers) {
                foreach ($neural_networks as $nn_type) {
                    if ($hidden_layers == 0 && $nn_type == "elman") continue;   //elman networks need at least 1 hidden layer

                    foreach ($bias_options as $use_bias) {
                        $bias_name = "";
                        if ($use_bias != "") $bias_name = "_b";

                        foreach ($search_types as $search_type) {

                            foreach ($population_sizes as $population_size) {
                                if ($search_type == "de") {
                                    foreach ($parent_selection_options as $parent_selection) {
                                        foreach ($recombination_options as $recombination) {
                                            foreach ($n_pairs_options as $n_pairs) {

                                                for ($task_name = 1; $task_name <= 20; $task_name++) {
                                                    start_de_job($flight_id, $output_target, $input_lags, $hidden_layers, $nn_type, $population_size, $parent_selection, $n_pairs, $recombination, $bias_name, $use_bias, $task_name);
                                                    $n_jobs++;
                                                }
                                            }
                                        }
                                    }

                                } else if ($search_type == "ps") {

                                    foreach ($inertia_weight_options as $inertia_weight) {
                                        foreach ($local_best_weight_options as $local_best_weight) {
                                            foreach ($global_best_weight_options as $global_best_weight) {

                                                for ($task_name = 1; $task_name <= 10; $task_name++) {
                                                    start_ps_job($flight_id, $output_target, $input_lags, $hidden_layers, $nn_type, $population_size, $inertia_weight, $local_best_weight, $global_best_weight, $bias_name, $use_bias, $task_name);
                                                    $n_jobs++;
                                                }
                                            }
                                        }
                                    }

                                } else {
                                    echo "UNKNOWN SEARCH TYPE: $search_type\n";
                                }
                            }

                        }
                    }
                }
            }
        }
    }
}

echo "created $n_jobs jobs.\n";

?>
