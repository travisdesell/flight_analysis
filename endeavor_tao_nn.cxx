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
#include "neural_networks/time_series_neural_network.hxx"

#include "mpi/mpi_ant_colony_optimization.hxx"
#include "mpi/mpi_particle_swarm.hxx"
#include "mpi/mpi_differential_evolution.hxx"

#include "asynchronous_algorithms/ant_colony_optimization.hxx"
#include "asynchronous_algorithms/neat.hxx"
#include "asynchronous_algorithms/particle_swarm.hxx"
#include "asynchronous_algorithms/differential_evolution.hxx"

#include "synchronous_algorithms/synchronous_newton_method.hxx"
#include "synchronous_algorithms/synchronous_gradient_descent.hxx"


//from undvc_common
#include "arguments.hxx"


TimeSeriesNeuralNetwork *ts_nn;

double objective_function(const vector<double> &parameters) {
    return ts_nn->objective_function(parameters);
}

vector<string> arguments;

double aco_objective_function(vector<Edge> &edges, vector<Edge> &recurrent_edges) {
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

    ts_nn->set_edges(edges, recurrent_edges);

    vector<double> min_bound(ts_nn->get_n_edges(), -1.5);
    vector<double> max_bound(ts_nn->get_n_edges(),  1.5);

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

double neat_objective_function(int n_hidden_layers, int nodes_per_layer, const vector<Edge> &edges, const vector<Edge> &recurrent_edges) {
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



int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, max_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &max_rank);

    arguments = vector<string>(argv, argv + argc);

    //read the flight data

    //make a vector of the parameters we're going to analyze with the NN
    //JIM SUGGESTS THESE:
    vector<string> column_headers;
    column_headers.push_back("ALT_STD");    //ALT_STD Pressure Alt    
    //column_headers.push_back("ALT_CPT");    //ALT_CPT Baro 1 Altimeter (inHg) 
    //column_headers.push_back("ALT_FO");     //ALT_FO  Baro 2 Altimeter (inHg) 
    //column_headers.push_back("ALT_SEL");    //ALT_SEL Selected altitude   
    column_headers.push_back("AOAL");
    column_headers.push_back("AOAR");
//    column_headers.push_back("FF1");        //fuel flow - engine 1
//    column_headers.push_back("FF2");        //fuel flow - engine 2
//    column_headers.push_back("GS");
    column_headers.push_back("IAS");        //indicated airspeed
    column_headers.push_back("ITT_1");      //interstage turbine temp - engine 1
    column_headers.push_back("ITT_2");      //interstage turbine temp - engine 2
    column_headers.push_back("IVV_R");      //vertical speed (feet per minute)
    column_headers.push_back("BLD_PRS1");   //bleed pressure (psi) - engine 1
    column_headers.push_back("BLD_PRS2");   //bleed pressure (psi) - engine 2
    column_headers.push_back("HYD_PRS1");   //hydrolic pressure valve fully closed - engine 1
    column_headers.push_back("HYD_PRS2");   //hydrolic pressure valve fully closed - engine 2
//    column_headers.push_back("N11");
//    column_headers.push_back("N12");
//    column_headers.push_back("N21");
//    column_headers.push_back("N22");
    column_headers.push_back("OIL_PRS_L");
    column_headers.push_back("OIL_PRS_R");
    column_headers.push_back("OIL_QTY1");
    column_headers.push_back("OIL_QTY2");
    column_headers.push_back("OIL_TMP1");
    column_headers.push_back("OIL_TMP2");
    column_headers.push_back("PITCH");
    column_headers.push_back("PITCH2");
    column_headers.push_back("PLA1");
    column_headers.push_back("PLA2");
    column_headers.push_back("ROLL");
    column_headers.push_back("ROLL_TRIM_P");
    column_headers.push_back("RUDD");
    column_headers.push_back("RUDD_TRIM_P");
    column_headers.push_back("SAT");
    column_headers.push_back("TAT");
    column_headers.push_back("VIB_N11");
    column_headers.push_back("VIB_N12");
    column_headers.push_back("VIB_N21");
    column_headers.push_back("VIB_N22");

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

    long seed;
    if (get_argument(arguments, "--seed", false, seed)) {
        srand48(seed);
    } else {
        srand48(time(NULL));
    }

    while (negative_flights.size() > 200) {
        negative_flights.erase( negative_flights.begin() + (negative_flights.size() * drand48()) );
    }

    read_flights(positive_flights, column_headers, n_positives, positives_rows, positives_columns, positives_data, true);
    read_flights(negative_flights, column_headers, n_negatives, negatives_rows, negatives_columns, negatives_data, true);


    cerr << "#";
    for (int i = 0; i < column_headers.size(); i++) cerr << " " << column_headers[i];
    cerr << endl;

    cout << "normalizing data sets" << endl;
    normalize_data_sets(column_headers, n_positives, positives_data, positives_rows, positives_columns, n_negatives, negatives_data, negatives_rows, negatives_columns);


    double **time_series_data = NULL;
    int time_series_rows = 0;
    int time_series_columns = 0;
    int output_target = 0;
    ts_nn = new TimeSeriesNeuralNetwork(output_target);
    ts_nn->set_time_series_data(time_series_data, time_series_rows, time_series_columns);

    string nn_filename;
    if (argument_exists(arguments, "--test_nn")) {
        get_argument(arguments, "--nn", false, nn_filename);
        //read the nn edges and weights from a file, then run it once
        ts_nn->read_nn_from_file(nn_filename);
        ts_nn->reset();

        double error = ts_nn->evaluate();
        cout << "total error: " << error << endl;
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


    } else if (argument_exists(arguments, "--aco_iterations")) {
        //get number ants
        //know input layer size already
        //get hidden layer size
        //get hidden layer nodes
        string aco_output_directory;
        int aco_iterations, n_ants, max_edge_pop_size, n_hidden_layers, nodes_per_layer;
        get_argument(arguments, "--aco_output_directory", true, aco_output_directory);
        get_argument(arguments, "--aco_iterations", true, aco_iterations);
        get_argument(arguments, "--n_ants", true, n_ants);
        get_argument(arguments, "--max_edge_pop_size", true, max_edge_pop_size);
        get_argument(arguments, "--n_hidden_layers", true, n_hidden_layers);
        get_argument(arguments, "--nodes_per_layer", true, nodes_per_layer);

        double pheromone_degradation_rate, pheromone_minimum, pheromone_maximum;
        get_argument(arguments, "--pheromone_degradation_rate", true, pheromone_degradation_rate);
        get_argument(arguments, "--pheromone_minimum", true, pheromone_minimum);
        get_argument(arguments, "--pheromone_maximum", true, pheromone_maximum);

        ts_nn->initialize_nodes(n_hidden_layers, nodes_per_layer);

        AntColony ant_colony(n_ants, max_edge_pop_size, time_series_columns, nodes_per_layer, n_hidden_layers, pheromone_degradation_rate, pheromone_minimum, pheromone_maximum);

        ant_colony.set_output_directory(aco_output_directory);
        ant_colony.set_compression(false);

        ant_colony_optimization_mpi(aco_iterations, ant_colony, aco_objective_function);

    } else {
        //read the nn edges from a file
        string nn_filename;
        get_argument(arguments, "--nn", true, nn_filename);
        ts_nn->read_nn_from_file(nn_filename);
        ts_nn->reset();

        vector<double> min_bound(ts_nn->get_n_edges(), -1.5);
        vector<double> max_bound(ts_nn->get_n_edges(),  1.5);

        if (rank == 0) {
            cout << "number of parameters: " << ts_nn->get_n_edges() << endl;
        }

        string search_type;
        get_argument(arguments, "--search_type", true, search_type);

        if (search_type.compare("ps") == 0) {
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
