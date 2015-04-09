#include <limits>
using std::numeric_limits;

#include <vector>
using std::vector;

#include <string>
using std::string;

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <iomanip>
using std::setw;
using std::setprecision;

#include <cmath>

#include "mpi.h"

#include "flight_io.hxx"

//from TAO
#include "neural_networks/edge.hxx"
#include "neural_networks/edge_new.hxx"
#include "neural_networks/neural_network.hxx"

#include "mpi/mpi_ant_colony_optimization_new.hxx"
#include "mpi/mpi_particle_swarm.hxx"
#include "mpi/mpi_differential_evolution.hxx"

#include "asynchronous_algorithms/ant_colony_optimization_new.hxx"
#include "asynchronous_algorithms/neat.hxx"
#include "asynchronous_algorithms/particle_swarm.hxx"
#include "asynchronous_algorithms/differential_evolution.hxx"

#include "synchronous_algorithms/synchronous_newton_method.hxx"
#include "synchronous_algorithms/synchronous_gradient_descent.hxx"


//from undvc_common
#include "arguments.hxx"


string search_type;
NeuralNetwork *nn;

double objective_function(const vector<double> &parameters) {
    return nn->objective_function(parameters);
}

vector<string> arguments;

double aco_objective_function(vector<EdgeNew> &edges, vector<EdgeNew> &recurrent_edges) {
    /*
    cout << "#feed forward edges" << endl;
    for (int j = 0; j < edges.size(); j++) {
        cout << edges[j] << endl;
    }
    cout << endl;

    cout << "#recurrent edges" << endl;
    for (int j = 0; j < recurrent_edges.size(); j++) {
        cout << recurrent_edges[j] << endl;
    }
    cout << endl;
    */

    //cout << "edges.size(): " << edges.size() << ", recurrent_edges.size(): " << recurrent_edges.size() << endl;

    nn->use_kahan_summation(true);
    nn->use_batch_update(true);

    nn->reset();
    nn->set_edges(edges, recurrent_edges);

    if (search_type.compare("backprop") == 0) {
        uint32_t backprop_iterations;
        get_argument(arguments, "--backprop_iterations", true, backprop_iterations);

        double learning_rate;
        get_argument(arguments, "--learning_rate", true, learning_rate);

        vector<double> starting_weights(nn->get_parameter_size(), 0.0);
        for (uint32_t i = 0; i < starting_weights.size(); i++) {
            starting_weights.at(i) = 0.05 * drand48();
        }

        return nn->backpropagation_time_series(starting_weights, learning_rate, backprop_iterations);
    } else {    //use PSO by default
        vector<double> min_bound(nn->get_parameter_size(), -1.5);
        vector<double> max_bound(nn->get_parameter_size(),  1.5);

        //cout << "parameter size: " << nn->get_parameter_size() << endl;

        ParticleSwarm ps(min_bound, max_bound, arguments);

        //run EA
        ps.iterate(objective_function);

        //set the weights using the best found individual  
        //dont need to set recurrent edges because they're weights are always 1
        vector<double> global_best = ps.get_global_best();
        int current = 0;
        for (int i = 0; i < edges.size(); i++) {
            edges[i].weight = global_best[current];
            current++;
        }

    /*
    cout << "#feed forward edges" << endl;
    for (int j = 0; j < edges.size(); j++) {
        cout << edges[j].src_layer << " " << edges[j].dst_layer << " " << edges[j].src_node << " " << edges[j].dst_node << " " << edges[j].weight << endl;
    }
    cout << endl;

    cout << "#recurrent edges" << endl;
    for (int j = 0; j < recurrent_edges.size(); j++) {
        cout << recurrent_edges[j].src_layer << " " << recurrent_edges[j].dst_layer << " " << recurrent_edges[j].src_node << " " << recurrent_edges[j].dst_node << " " << recurrent_edges[j].weight << endl;
    }
    cout << endl;
    */

    /*
    cout << "global best.size: " << global_best.size() << endl;
    cout << "edges.size: " << edges.size() << endl;
    cout << "recurrent_edges.size: " << recurrent_edges.size() << endl;
    */

        return ps.get_global_best_fitness();
    }
}

//double neat_objective_function(int n_hidden_layers, int nodes_per_layer, const vector<Edge> &edges, const vector<Edge> &recurrent_edges) {
    /*
    cout << "#feed forward edges" << endl;
    for (int j = 0; j < edges.size(); j++) {
        cout << edges[j] << endl;
    }
    cout << endl;

    cout << "#recurrent edges" << endl;
    for (int j = 0; j < recurrent_edges.size(); j++) {
        cout << recurrent_edges[j] << endl;
    }
    cout << endl;

    cout << "n_hidden_layers: " << n_hidden_layers << ", nodes_per_layer: " << nodes_per_layer << endl;
    */

/*
    ts_nn->initialize_nodes(n_hidden_layers, nodes_per_layer);
    ts_nn->set_edges(edges, recurrent_edges);

    //double fitness = ts_nn->objective_function();

    //don't need to do recurrent edges as they always have a weight of 1
    vector<double> starting_point;
    for (int i = 0; i < edges.size(); i++) {
        starting_point.push_back(edges[i].weight);
    }
    vector<double> step_size( starting_point.size(), 0.0001 );


    double fitness;
    vector<double> final_parameters;
    synchronous_gradient_descent(arguments, objective_function, starting_point, step_size, final_parameters, fitness);

//    cout << "n_hidden_layers: " << n_hidden_layers << ", nodes per layer: " << nodes_per_layer << ", starting_point.size(): " << starting_point.size() << ", step_size.size(): " << step_size.size() << ", fitness: " << fitness << endl;

    return fitness;
}
*/


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, max_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &max_rank);

    arguments = vector<string>(argv, argv + argc);

    long seed;
    if (get_argument(arguments, "--seed", false, seed)) {
        srand48(seed);
    } else {
        srand48(time(NULL));
    }

    string endeavor_headers[] = {"ALT_STD", "AOAL", "AOAR", "IAS", "ITT_1", "ITT_2", "IVV_R", "BLD_PRS1", "BLD_PRS2", "HYD_PRS1", "HYD_PRS2", "OIL_PRS_L", "OIL_PRS_R", "OIL_QTY1", "OIL_QTY2", "OIL_TMP1", "OIL_TMP2", "PITCH", "PITCH2", "PLA1", "PLA2", "ROLL", "ROLL_TRIM_P", "RUDD", "RUDD_TRIM_P", "SAT", "TAT", "VIB_N11", "VIB_N12", "VIB_N21", "VIB_N22"};

    string ngafid_headers[] = {"indicated_airspeed", "msl_altitude", "pitch_attitude", "roll_attitude"};

    vector<string> column_headers;
    
    bool endeavor_data;
    if (argument_exists(arguments, "--endeavor")) {
        endeavor_data = true;
        column_headers = vector<string>(endeavor_headers, endeavor_headers + 31);
    } else {
        endeavor_data = false;
        column_headers = vector<string>(ngafid_headers, ngafid_headers + 4);
    }

    string flight_filename;
    get_argument(arguments, "--flight_file", true, flight_filename);

    uint32_t rows, cols;
    double **flight_data = NULL;

    read_flight_file(flight_filename, column_headers, rows, cols, flight_data, endeavor_data);

    if (argument_exists(arguments, "--normalize_data")) {
        //normalize the data
        vector<double> mins(cols, numeric_limits<double>::max());
        vector<double> maxs(cols, -numeric_limits<double>::max());

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (flight_data[i][j] < mins[j]) mins[j] = flight_data[i][j];
                if (flight_data[i][j] > maxs[j]) maxs[j] = flight_data[i][j];
            }   
        }   

        if (rank == 0) {
            cerr << "#headers: ";
            for (int i = 0; i < cols; i++) cerr << " " << column_headers[i];
            cerr << endl;

            cerr << "#minimums: ";
            for (int i = 0; i < cols; i++) cerr << " " << mins[i];
            cerr << endl;

            cerr << "#maximums: ";
            for (int i = 0; i < cols; i++) cerr << " " << maxs[i];
            cerr << endl;
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (0 == column_headers[j].compare("roll_attitude")) {
                    flight_data[i][j] = flight_data[i][j] / 60; 

                } else if (0 == column_headers[j].compare("pitch_attitude")) {
                    flight_data[i][j] = flight_data[i][j] / 60; 

                } else if (0 == column_headers[j].compare("indicated_airspeed")) {
                    flight_data[i][j] = flight_data[i][j] / 200;

                } else if (0 == column_headers[j].compare("msl_altitude")) {
                    flight_data[i][j] = flight_data[i][j] / 6000;

                } else {
                    flight_data[i][j] = (flight_data[i][j] - mins[j]) / (maxs[j] - mins[j]);
                }   
            }   
        }   
    }

    if (argument_exists(arguments, "--print_flight_data")) {
        cout << "#";
        for (int i = 0; i < column_headers.size(); i++) cout << " " << column_headers[i];
        cout << endl;

        for (uint32_t i = 0; i < rows; i++) {
            for (uint32_t j = 0; j < cols; j++) {
                cout << " " << flight_data[i][j];
            }
            cout << endl;
        }
    }


    vector<string> output_headers;
    get_argument_vector(arguments, "--output_headers", true, output_headers);

    double **output_data;
    get_output_data(flight_data, rows, cols, column_headers, output_headers, &output_data);

    if (argument_exists(arguments, "--print_output_data")) {
        cout << "#";
        for (int i = 0; i < output_headers.size(); i++) cout << " " << output_headers[i];
        cout << endl;

        for (uint32_t i = 0; i < rows - 1; i++) {
            for (uint32_t j = 0; j < output_headers.size(); j++) {
                cout << " " << output_data[i][j];
            }
            cout << endl;
        }
    }


    /*
    int n_positives = 0, n_negatives = 0;
    vector<string> positive_flights, negative_flights;
    vector<uint32_t> positives_rows, negatives_rows;
    vector<uint32_t> positives_columns, negatives_columns;

    double ***positives_data = NULL;
    double ***negatives_data = NULL;

    string positives_filename, negatives_filename;
    if (get_argument(arguments, "--positives_file", false, positives_filename)) {
        get_argument(arguments, "--negatives_file", true, negatives_filename);

        get_flight_files_from_file(positives_filename, positive_flights);
        get_flight_files_from_file(negatives_filename, negative_flights);

    } else {
        string positives_dir, negatives_dir;
        get_argument(arguments, "--positives_dir", true, positives_dir);
        get_argument(arguments, "--negatives_dir", true, negatives_dir);

        get_flight_files_from_directory(positives_dir, positive_flights);
        get_flight_files_from_directory(negatives_dir, negative_flights);
    }

    while (negative_flights.size() > 200) {
        negative_flights.erase( negative_flights.begin() + (negative_flights.size() * drand48()) );
    }

    read_flights(positive_flights, column_headers, n_positives, positives_rows, positives_columns, positives_data, endeavor_data);
    read_flights(negative_flights, column_headers, n_negatives, negatives_rows, negatives_columns, negatives_data, endeavor_data);

    cerr << "#";
    for (int i = 0; i < column_headers.size(); i++) cerr << " " << column_headers[i];
    cerr << endl;

    cout << "normalizing data sets" << endl;
    normalize_data_sets(column_headers, n_positives, positives_data, positives_rows, positives_columns, n_negatives, negatives_data, negatives_rows, negatives_columns);

    */

    string nn_filename;
    /*
    if (argument_exists(arguments, "--test_nn")) {
        get_argument(arguments, "--nn", false, nn_filename);
        //read the nn edges and weights from a file, then run it once
        nn = new NeuralNetwork(nn_filename);

        double error = nn->objective_function();
        cout << "total error: " << error << endl;
        */
    /*
    } else if (argument_exists(arguments, "--neat_iterations")) {

        int neat_iterations;
        get_argument(arguments, "--neat_iterations", true, neat_iterations);

        //NEAT parameters
        double excess_weight = 1.0;
        double disjoint_weight = 1.0;
        double weight_weight = 1.0;
        double compatibility_threshold = 2.0;
        double normalization = 1.0;

        double mutation_without_crossover_rate = 0.25; 
        double add_node_mutation_rate = 0.01; 
        double add_link_mutation_rate = 0.1; 
        double interspecies_crossover_rate = 0.05;
        double crossover_weight_average_rate = 0.4;

        double weight_mutation_rate = 0.8; 
        double random_weight_mutation_rate = 0.1;
        double uniform_weight_mutation_rate = 0.9;
        double uniform_perturbation = 0.001;

        //NEED:
        double enable_if_both_parents_disabled = 0.25;

        int population_size = 100;

        NEAT neat(excess_weight, disjoint_weight, weight_weight, compatibility_threshold, normalization,
                  mutation_without_crossover_rate, weight_mutation_rate, add_node_mutation_rate,
                  add_link_mutation_rate, interspecies_crossover_rate, crossover_weight_average_rate, random_weight_mutation_rate,
                  uniform_weight_mutation_rate, uniform_perturbation, enable_if_both_parents_disabled, population_size);

        int n_input_nodes = time_series_columns;
        int n_output_nodes = 1;
        neat.iterate(neat_iterations, n_input_nodes, n_output_nodes, neat_objective_function);
    */



    double pheromone_placement_rate = 1.0;
    double pheromone_degradation_rate = 0.95;
    double maximum_pheromone = 10.0;
    double minimum_pheromone = 0.1;
    uint32_t number_ants = 12;
    uint32_t recurrent_depth = 2;
    uint32_t n_input_nodes = cols;
    uint32_t n_hidden_layers = 3;
    uint32_t n_hidden_nodes = n_input_nodes;
    uint32_t n_output_nodes = output_headers.size();

    get_argument(arguments, "--search_type", true, search_type);

    if (argument_exists(arguments, "--aco_iterations")) {
        //string aco_output_directory;
        uint32_t aco_iterations;
        //get_argument(arguments, "--aco_output_directory", true, aco_output_directory);
        get_argument(arguments, "--aco_iterations", true, aco_iterations);

        uint32_t aco_population_size;
        get_argument(arguments, "--aco_population_size", true, aco_population_size);

        /*
        uint32_t number_ants;
        double pheromone_placement_rate, pheromone_degradation_rate, minimum_pheromone, maximum_pheromone;
        get_argument(arguments, "--pheromone_placement_rate", true, pheromone_placement_rate);
        get_argument(arguments, "--pheromone_degradation_rate", true, pheromone_degradation_rate);
        get_argument(arguments, "--minimum_pheromone", true, minimum_pheromone);
        get_argument(arguments, "--maximum_pheromone", true, maximum_pheromone);
        get_argument(arguments, "--number_ants", true, number_ants);
        */

        nn = new NeuralNetwork(recurrent_depth, n_input_nodes, n_hidden_layers, n_hidden_nodes, n_output_nodes, "linear");

        nn->set_training_data(rows - 1, cols, flight_data, 1, output_data);


        AntColonyNew ant_colony(pheromone_placement_rate, pheromone_degradation_rate, maximum_pheromone, minimum_pheromone, number_ants, recurrent_depth, n_input_nodes, n_hidden_layers, n_hidden_nodes, n_output_nodes, aco_population_size);

        //ant_colony.set_output_directory(aco_output_directory);

        ant_colony_optimization_mpi(aco_iterations, ant_colony, aco_objective_function);

    } else {
        //read the nn edges from a file
        string nn_filename;

        if (argument_exists(arguments, "--nn")) {
            get_argument(arguments, "--nn", true, nn_filename);
            nn = new NeuralNetwork(nn_filename);
        } else {
            recurrent_depth = 2;
            n_hidden_layers = 1;
            n_hidden_nodes = 4;
            n_output_nodes = 1;
            nn = new NeuralNetwork(recurrent_depth, n_input_nodes, n_hidden_layers, n_hidden_nodes, n_output_nodes, "linear");

            nn->set_training_data(rows - 1, cols, flight_data, 1, output_data);

            vector<EdgeNew> edges;
            vector<EdgeNew> recurrent_edges;

            // THIS IS A JORDAN NETWORK
            edges.push_back(EdgeNew(0, 0, 0, 0, 1, 0));
            edges.push_back(EdgeNew(0, 0, 0, 0, 1, 1));
            edges.push_back(EdgeNew(0, 0, 0, 0, 1, 2));
            edges.push_back(EdgeNew(0, 0, 0, 0, 1, 3));

            edges.push_back(EdgeNew(0, 0, 1, 0, 1, 0));
            edges.push_back(EdgeNew(0, 0, 1, 0, 1, 1));
            edges.push_back(EdgeNew(0, 0, 1, 0, 1, 2));
            edges.push_back(EdgeNew(0, 0, 1, 0, 1, 3));

            edges.push_back(EdgeNew(0, 0, 2, 0, 1, 0));
            edges.push_back(EdgeNew(0, 0, 2, 0, 1, 1));
            edges.push_back(EdgeNew(0, 0, 2, 0, 1, 2));
            edges.push_back(EdgeNew(0, 0, 2, 0, 1, 3));

            edges.push_back(EdgeNew(0, 0, 3, 0, 1, 0));
            edges.push_back(EdgeNew(0, 0, 3, 0, 1, 1));
            edges.push_back(EdgeNew(0, 0, 3, 0, 1, 2));
            edges.push_back(EdgeNew(0, 0, 3, 0, 1, 3));

            edges.push_back(EdgeNew(1, 0, 0, 0, 1, 0));
            edges.push_back(EdgeNew(1, 0, 0, 0, 1, 1));
            edges.push_back(EdgeNew(1, 0, 0, 0, 1, 2));
            edges.push_back(EdgeNew(1, 0, 0, 0, 1, 3));

            edges.push_back(EdgeNew(0, 1, 0, 0, 2, 0));
            edges.push_back(EdgeNew(0, 1, 1, 0, 2, 0));
            edges.push_back(EdgeNew(0, 1, 2, 0, 2, 0));
            edges.push_back(EdgeNew(0, 1, 3, 0, 2, 0));

            recurrent_edges.push_back(EdgeNew(0, 2, 0, 1, 0, 0));

            nn->set_edges(edges, recurrent_edges);
        }

        vector<double> min_bound(nn->get_parameter_size(), -1.5);
        vector<double> max_bound(nn->get_parameter_size(),  1.5);

        if (rank == 0) {
            cout << "number of parameters: " << nn->get_parameter_size() << endl;
        }

        if (search_type.compare("backprop") == 0) {
            uint32_t backprop_iterations;
            get_argument(arguments, "--backprop_iterations", true, backprop_iterations);

            double learning_rate;
            get_argument(arguments, "--learning_rate", true, learning_rate);

            vector<double> starting_weights(nn->get_parameter_size(), 0.0);
            for (uint32_t i = 0; i < starting_weights.size(); i++) {
                starting_weights.at(i) = 0.05 * drand48();
            }

            cout << nn->json() << endl;

            nn->use_kahan_summation(true);
            nn->use_batch_update(true);

            nn->backpropagation_time_series(starting_weights, learning_rate, backprop_iterations);
        } else if (search_type.compare("ps") == 0) {
            ParticleSwarm ps(min_bound, max_bound, arguments);
            ps.iterate(objective_function);

        } else if (search_type.compare("de") == 0) {
            DifferentialEvolution de(min_bound, max_bound, arguments);
            de.iterate(objective_function);

        } else if (search_type.compare("ps_mpi") == 0) {
            ParticleSwarmMPI ps(min_bound, max_bound, arguments);
            ps.go(objective_function);

        } else if (search_type.compare("de_mpi") == 0) {
            DifferentialEvolutionMPI de(min_bound, max_bound, arguments);
            de.go(objective_function);

        } else if (search_type.compare("snm") == 0 || search_type.compare("gd") == 0 || search_type.compare("cgd") == 0) {
            string starting_point_s;
            vector<double> starting_point(min_bound.size(), 0);

            if (get_argument(arguments, "--starting_point", false, starting_point_s)) {
                cout << "#starting point: '" << starting_point_s << "'" << endl;
                string_to_vector(starting_point_s, starting_point);
            } else {
                for (unsigned int i = 0; i < min_bound.size(); i++) {
                    starting_point[i] = min_bound[i] + ((max_bound[i] - min_bound[i]) * drand48());
                }
            }

            vector<double> step_size(min_bound.size(), 0.001);

            if (search_type.compare("snm") == 0) {
                synchronous_newton_method(arguments, objective_function, starting_point, step_size);
            } else if (search_type.compare("gd") == 0) {
                vector<double> final_parameters;
                double final_fitness;
                synchronous_gradient_descent(arguments, objective_function, starting_point, step_size, final_parameters, final_fitness);
            } else if (search_type.compare("cgd") == 0) {
                synchronous_conjugate_gradient_descent(arguments, objective_function, starting_point, step_size);
            }

        } else {
            fprintf(stderr, "Improperly specified search type: '%s'\n", search_type.c_str());
            fprintf(stderr, "Possibilities are:\n");
            fprintf(stderr, "    de     -       differential evolution\n");
            fprintf(stderr, "    ps     -       particle swarm optimization\n");
            fprintf(stderr, "    de_mpi -       asynchronous differential evolution over MPI\n");
            fprintf(stderr, "    ps_mpi -       asynchronous particle swarm optimization over MPI\n");
            fprintf(stderr, "    snm    -       synchronous newton method\n");
            fprintf(stderr, "    gd     -       gradient descent\n");
            fprintf(stderr, "    cgd    -       conjugate gradient descent\n");

            exit(0);
        }
    }
}
