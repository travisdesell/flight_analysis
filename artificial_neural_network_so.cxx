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

#include <cmath>

#include "mpi.h"

#include "flight_io.hxx"

#include "mpi/mpi_particle_swarm.hxx"
#include "mpi/mpi_differential_evolution.hxx"

#include "asynchronous_algorithms/particle_swarm.hxx"
#include "asynchronous_algorithms/differential_evolution.hxx"

#include "tao/synchronous_algorithms/synchronous_newton_method.hxx"
#include "tao/synchronous_algorithms/synchronous_gradient_descent.hxx"


//from undvc_common
#include "undvc_common/arguments.hxx"

#define PRECISION double

class ArtificialNeuralNetwork {
    private:
        const unsigned int input_layer_size;
        const unsigned int hidden_layers;
        const unsigned int hidden_layer_size;
        const unsigned int output_layer_size;
        const unsigned int recurrent_layer_size;

        const bool use_bias;

        const unsigned int type;

        PRECISION *hidden_layer;
        PRECISION *output_layer;
        PRECISION *recurrent_layer;

    public:
        const static unsigned int FEED_FORWARD_NETWORK = 0;
        const static unsigned int ELMAN_NETWORK = 1;
        const static unsigned int JORDAN_NETWORK = 2;

        PRECISION get_output_layer(int i) {
            return output_layer[i];
        }

        ArtificialNeuralNetwork(unsigned int ils,
                                unsigned int hl,
                                unsigned int hls,
                                unsigned int ols,
                                unsigned int rls,
                                bool ub,
                                unsigned int t) : 
                                input_layer_size(ils),
                                hidden_layers(hl),
                                hidden_layer_size(hls),
                                output_layer_size(ols),
                                recurrent_layer_size(rls),
                                use_bias(ub),
                                type(t) {

            cout << "RECURRENT_LAYER_SIZE: " << recurrent_layer_size << endl;

            hidden_layer = new PRECISION[hidden_layer_size * hidden_layers];
            output_layer = new PRECISION[output_layer_size];

            if (type != FEED_FORWARD_NETWORK) recurrent_layer = new PRECISION[recurrent_layer_size];
            reset();
        }

        void reset() {
            if (type != FEED_FORWARD_NETWORK) {
                for (unsigned int i = 0; i < recurrent_layer_size; i++) {
                    recurrent_layer[i] = 0;
                }
            }
        }

        PRECISION evaluate(const vector<PRECISION> &weights, const double *input_layer, const double *expected_output) {
            unsigned int current_weight = 0;

            if (hidden_layers == 0) {
                for (unsigned int i = 0; i < output_layer_size; i++) {
                    output_layer[i] = 0;

                    for (unsigned int j = 0; j < input_layer_size; j++) {
//                        cout << "weights[" << current_weight << "] " << weights[current_weight] << " * input_layer[" << j << "] " << input_layer[j] << endl;

                        output_layer[i] += weights[current_weight] * input_layer[j];

//                        if (output_layer[i] >  1) output_layer[i] =  1;
//                        if (output_layer[i] < -1) output_layer[i] = -1;
                        current_weight++;
                    }

//                    cout << "output_layer[" << i << "]: " << output_layer[i] << endl;
                }

                if (type == JORDAN_NETWORK) {   //can have a jordan network without a hidden layer
                    for (unsigned int i = 0; i < output_layer_size; i++) {
                        for (unsigned int j = 0; j < recurrent_layer_size; j++) {
                            output_layer[i] += weights[current_weight] * recurrent_layer[j];
                            current_weight++;
                        }
                    }
                }

            } else {    //hidden layers > 0
                //Apply weights and inputs to first hidden layer
                for (unsigned int i = 0; i < hidden_layer_size; i++) {
                    if (use_bias) {
                        hidden_layer[i] += weights[current_weight];
//                        if (hidden_layer[i] >  1) hidden_layer[i] =  1;
//                        if (hidden_layer[i] < -1) hidden_layer[i] = -1;

                        current_weight++;
                    } else {
                        hidden_layer[i] = 0;
                    }

                    for (unsigned int j = 0; j < input_layer_size; j++) {
                        hidden_layer[i] += weights[current_weight] * input_layer[j];
                        current_weight++;
                    }
                }

                if (type != FEED_FORWARD_NETWORK) {
                    //Apply the recurrent layer to the first hidden layer (if there is a recurrent layer)
                    for (unsigned int i = 0; i < hidden_layer_size; i++) {
                        for (unsigned int j = 0; j < recurrent_layer_size; j++) {
                            hidden_layer[i] += weights[current_weight] * recurrent_layer[j];
//                            if (hidden_layer[i] >  1) hidden_layer[i] =  1;
//                            if (hidden_layer[i] < -1) hidden_layer[i] = -1;

                            current_weight++;
                        }
                    }
                }

                for (unsigned int i = 1; i < hidden_layers; i++) {
                    for (unsigned int j = 0; j < hidden_layer_size; j++) {
                        if (use_bias) {
                            hidden_layer[(i * hidden_layer_size) + j] = weights[current_weight];
//                            if (hidden_layer[(i * hidden_layer_size) + j] >  1) hidden_layer[i] =  1;
//                            if (hidden_layer[(i * hidden_layer_size) + j] < -1) hidden_layer[i] = -1;

                            current_weight++;
                        } else {
                            hidden_layer[(i * hidden_layer_size) + j] = 0;
                        }

                        for (unsigned int k = 0; k < hidden_layer_size; k++) {
                            hidden_layer[(i * hidden_layer_size) + j] += weights[current_weight] * hidden_layer[((i - 1) * hidden_layer_size) + j];
//                            if (hidden_layer[(i * hidden_layer_size) + j] >  1) hidden_layer[i] =  1;
//                            if (hidden_layer[(i * hidden_layer_size) + j] < -1) hidden_layer[i] = -1;

                            current_weight++;
                        }
                    }
                }

                for (unsigned int i = 0; i < output_layer_size; i++) {
                    output_layer[i] = 0;

                    for (unsigned int j = 0; j < hidden_layer_size; j++) {
                        output_layer[i] += weights[current_weight] * hidden_layer[((hidden_layers - 1) * hidden_layer_size) + j];
//                        if (output_layer[i] >  1) output_layer[i] =  1;
//                        if (output_layer[i] < -1) output_layer[i] = -1;

                        current_weight++;
                    }
                }
            }

            if (use_bias) {
                for (unsigned int i = 0; i < output_layer_size; i++) {
                    output_layer[i] += weights[current_weight];
//                    if (output_layer[i] >  1) output_layer[i] =  1;
//                    if (output_layer[i] < -1) output_layer[i] = -1;

                    current_weight++;
                }
            }

            if (current_weight != weights.size()) {
                cout << "current_weight: " << current_weight << ", weights.size(): " << weights.size() << endl;
                exit(0);
            }

            //Update the recurrent layer.
            //Need to update for mutliple hidden layers
            if (type == ELMAN_NETWORK) {
                for (unsigned int i = 0; i < hidden_layer_size; i++) {
                    recurrent_layer[i] = hidden_layer[i];
                    if (recurrent_layer[i] > 3) recurrent_layer[i] = 3;
                    if (recurrent_layer[i] < -2) recurrent_layer[i] = -2;
                }
            } else if (type == JORDAN_NETWORK) {
                for (unsigned int i = 0; i < output_layer_size; i++) {
                    recurrent_layer[i] = output_layer[i];
                    if (recurrent_layer[i] > 3) recurrent_layer[i] = 3;
                    if (recurrent_layer[i] < -2) recurrent_layer[i] = -2;
                }
            }

            PRECISION error = 0.0;
            PRECISION temp;
//            PRECISION max_error = 0;
            for (unsigned int i = 0; i < output_layer_size; i++) {
                temp = output_layer[i] - expected_output[i];
//                if (fabs(temp) > max_error) max_error = fabs(temp);
//                error += temp * temp;
                error += fabs(temp);
            }

            return error / output_layer_size;
//            return max_error;
        }

};

unsigned int output_target;
int seconds_into_future = 0;
double* flight_data = NULL;
double* flight_data_delta = NULL;
double* flight_data_delta2 = NULL;
unsigned int flight_rows;
unsigned int flight_columns;
unsigned int input_lags = 0;
int output_timesteps;
ArtificialNeuralNetwork *ann;

double *input_data;

double objective_function(const vector<double> &parameters) {
    double total_error = 0;
    double current_error;
//    double max_error = 0;
    
    ann->reset();

    for (unsigned int i = 0; i < flight_rows - (input_lags + 1 + output_timesteps + seconds_into_future); i++) {

        for (unsigned int j = 0; j < flight_columns; j++) input_data[j] = flight_data[(i * flight_columns) + j];
        if (input_lags > 0) {
            for (unsigned int j = 0; j < flight_columns; j++) input_data[j + flight_columns] = flight_data_delta[(i * flight_columns) + j];
        }
        
        if (input_lags > 1) {
            for (unsigned int j = 0; j < flight_columns; j++) input_data[j + flight_columns + flight_columns] = flight_data_delta2[(i * flight_columns) + j];
        }

        /*
        for (int k = 0; k < (input_lags + 1) * flight_columns; k++) {
            cout << " " << input_data[k];
        }
        cout << endl;
        */

        current_error = ann->evaluate( parameters, input_data, &(flight_data[((i + 1 + seconds_into_future) * flight_columns) + output_target]) );

//        cout << setw(15) << total_error << " - " << setw(10) << current_error << " - " << max_error << endl;
        total_error += (current_error / flight_rows);
//        if (current_error > max_error) max_error = current_error;
    }

    /*
    if (total_error != total_error) {
        cerr << "ERROR! total error is not a number!" << endl;
        cout << "ERROR! total error is not a number!" << endl;
        exit(0);
    }

    if (isinf(total_error)) {
        cerr << "ERROR! total error is infinity!" << endl;
        cout << "ERROR! total error is infinity!" << endl;
        exit(0);
    }
    */


//    cout << "total_error: " << total_error << endl;
    return -total_error;
//    return -(sqrt(total_error) / flight_rows);
//    cout << "max_error: " << max_error << endl;
//    return -max_error;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, max_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &max_rank);

    //cout << "process: " << rank << " of " << max_rank << endl;

    vector<string> arguments(argv, argv + argc);
    
    vector<string> column_headers;

    string input_filename;
    get_argument(arguments, "--input_filename", true, input_filename);
    read_flight_file(input_filename, flight_rows, flight_columns, flight_data, column_headers);

    //use these to store values for pre-calculated delta (difference between previous 
    //and current value) and delta^2 (difference between the previous two deltas)
    flight_data_delta = new double[flight_columns * flight_rows];
    flight_data_delta2 = new double[flight_columns * flight_rows];

    /*
    for (int i = 0; i < flight_rows; i++) {
        for (int j = 0; j < flight_columns; j++) {
            cout << setw(10) << flight_data[(i * flight_columns) + j];
        }
        cout << endl;
    }
    */

    //Normalize the flight data -- 1. get min/max for each value
    vector<double> min(flight_columns, 0), max(flight_columns, 0);
    for (unsigned int j = 0; j < flight_columns; j++) {

        min[j] = flight_data[j];
        max[j] = flight_data[j];

        for (unsigned int i = 1; i < flight_rows; i++) {
            double current = flight_data[(i * flight_columns) + j];
            if (current > max[j]) max[j] = current;
            if (current < min[j]) min[j] = current;
        }
    }

    //Normalize the flight data -- 2. update the values
    vector<double> min_delta(flight_columns, 0), max_delta(flight_columns, 0);
    vector<double> min_delta2(flight_columns, 0), max_delta2(flight_columns, 0);
    vector<double> avg_delta(flight_columns, 0);
    vector<double> avg_delta2(flight_columns, 0);

    for (unsigned int j = 0; j < flight_columns; j++) {
        for (unsigned int i = 0; i < flight_rows; i++) {
            double current = flight_data[(i * flight_columns) + j];

            flight_data[(i * flight_columns) + j] = (current - min[j]) / (max[j] - min[j]);

            double current_delta, current_delta2;
            if (i == 0) {
                current_delta = 0;
                current_delta2 = 0;

                flight_data_delta[(i * flight_columns) + j] = 0;
                flight_data_delta2[(i * flight_columns) + j] = 0;

            } else if (i == 1) {
                current_delta = flight_data[(i * flight_columns) + j] - flight_data[((i-1) * flight_columns) + j];
                current_delta2 = 0;

                flight_data_delta[(i * flight_columns) + j] = current_delta;
                flight_data_delta2[(i * flight_columns) + j] = 0;
            } else {
                current_delta = flight_data[(i * flight_columns) + j] - flight_data[((i-1) * flight_columns) + j];
                current_delta2 = current_delta - flight_data_delta[((i-1) * flight_columns) + j];

                flight_data_delta[(i * flight_columns) + j] = current_delta;
                flight_data_delta2[(i * flight_columns) + j] = current_delta2;
            }

            if (current_delta > max_delta[j]) max_delta[j] = current_delta;
            if (current_delta < min_delta[j]) min_delta[j] = current_delta;
            if (current_delta2 > max_delta2[j]) max_delta2[j] = current_delta2;
            if (current_delta2 < min_delta2[j]) min_delta2[j] = current_delta2;

            avg_delta[j] += fabs(current_delta);
            avg_delta2[j] += fabs(current_delta2);
        }
    }

    for (unsigned int j = 0; j < flight_columns; j++) {
        avg_delta[j] = avg_delta[j] / (flight_rows - 1);
        avg_delta2[j] = avg_delta2[j] / (flight_rows - 2);
    }

    if (rank == 0) {
        cout << "#min:       ";
        for (unsigned int i = 0; i < flight_columns; i++) cout << " " << setw(20) << min[i];
        cout << endl;

        cout << "#max:       ";
        for (unsigned int i = 0; i < flight_columns; i++) cout << " " << setw(20) << max[i];
        cout << endl;

        cout << "#min_delta: ";
        for (unsigned int i = 0; i < flight_columns; i++) cout << " " << setw(20) << min_delta[i];
        cout << endl;

        cout << "#max_delta: ";
        for (unsigned int i = 0; i < flight_columns; i++) cout << " " << setw(20) << max_delta2[i];
        cout << endl;

        cout << "#avg_delta: ";
        for (unsigned int i = 0; i < flight_columns; i++) cout << " " << setw(20) << avg_delta2[i];
        cout << endl;

        cout << "#min_delta2:";
        for (unsigned int i = 0; i < flight_columns; i++) cout << " " << setw(20) << min_delta2[i];
        cout << endl;

        cout << "#max_delta2:";
        for (unsigned int i = 0; i < flight_columns; i++) cout << " " << setw(20) << max_delta2[i];
        cout << endl;

        cout << "#avg_delta2:";
        for (unsigned int i = 0; i < flight_columns; i++) cout << " " << setw(20) << avg_delta2[i];
        cout << endl;
    }

    double err_prev = 0, err_delta = 0, err_delta2 = 0;

    for (unsigned int i = 0; i < flight_rows; i++) {
        double prev_predict = 0, delta_predict = 0, delta2_predict = 0;

        for (unsigned int j = 0; j < flight_columns; j++) {
            //prediction is the previous value
            if (i > 0) {
                double p = flight_data[(i * flight_columns) + j] - flight_data[((i - 1) * flight_columns) + j];
//                prev_predict += p * p;
                prev_predict += fabs(p);
            }

            //prediction is the previous value plus the previous delta
            if (i > 1) {
                double p = flight_data[(i * flight_columns) + j] - (flight_data[((i - 1) * flight_columns) + j] + flight_data_delta[((i - 1) * flight_columns) + j]);
//                delta_predict += p * p;
                delta_predict += fabs(p);
            }

            //prediction is the previous value plus the previous delta plus the change in previous delta
            if (i > 2) {
                double p = flight_data[(i * flight_columns) + j] - (flight_data[((i - 1) * flight_columns) + j] + flight_data_delta[((i - 1) * flight_columns) + j] + flight_data_delta2[((i - 1) * flight_columns) + j]);
//                delta2_predict += p * p;
                delta2_predict += fabs(p);
            }
        }

        err_prev   += prev_predict / 4;
        err_delta  += delta_predict / 4;
        err_delta2 += delta2_predict / 4;
    }

    /*
    err_prev   = sqrt(err_prev)   /  flight_rows;
    err_delta  = sqrt(err_delta2) / (flight_rows - 1);
    err_delta2 = sqrt(err_delta2) / (flight_rows - 2);
    */

    err_prev   = err_prev   /  flight_rows;
    err_delta  = err_delta2 / (flight_rows - 1);
    err_delta2 = err_delta2 / (flight_rows - 2);

    if (rank == 0) {
        cout << "#err prev:   " << err_prev << endl;
        cout << "#err delta:  " << err_delta << endl;
        cout << "#err delta2: " << err_delta2 << endl;
    }

    /*
    for (unsigned int i = 0; i < flight_rows; i++) {
        for (unsigned int j = 0; j < flight_columns; j++) {
            unsigned int pos = (i * flight_columns) + j;

            cout << "     |" << setw(20) << flight_data[pos] << setw(20) << flight_data_delta[pos] << setw(20) << flight_data_delta2[pos];
        }

        cout << "     |" << endl;
    }
    */


    //determine how many previous timesteps will be fed into the neural network
    get_argument(arguments, "--input_lags", false, input_lags);
    unsigned int input_layer_size = (input_lags + 1) * flight_columns;
    input_data = new double[input_layer_size];

    string output_str;
    get_argument(arguments, "--output_target", true, output_str);
    output_target = -1;
    if (0 == output_str.compare("roll")) {
        output_target = 0;
    } else if (0 == output_str.compare("pitch")) {
        output_target = 1;
    } else if (0 == output_str.compare("airspeed")) {
        output_target = 2;
    } else if (0 == output_str.compare("altitude")) {
        output_target = 3;
    } else {
        cerr << "Error, misspecified output target '" << output_str << "', possibilities:" << endl;
        cerr << "    airspeed" << endl;
        cerr << "    roll" << endl;
        cerr << "    pitch" << endl;
        cerr << "    altitude" << endl;
        exit(1);
    }


    unsigned int hidden_layers;
    get_argument(arguments, "--hidden_layers", true, hidden_layers);

    string network_type_s;
    unsigned int network_type;
    get_argument(arguments, "--network_type", true, network_type_s);

    unsigned int recurrent_layers = 0;
    if (0 == network_type_s.compare("feed_forward"))    network_type = ArtificialNeuralNetwork::FEED_FORWARD_NETWORK;
    else if (0 == network_type_s.compare("elman"))      network_type = ArtificialNeuralNetwork::ELMAN_NETWORK;
    else if (0 == network_type_s.compare("jordan"))     network_type = ArtificialNeuralNetwork::JORDAN_NETWORK;
    else {
        cerr << "Unknown 'network_type' argument, possibilities:" << endl;
        cerr << "    feed_forward" << endl;
        cerr << "    elman" << endl;
        cerr << "    jordan" << endl;
        exit(0);
    }

    unsigned int recurrent_layer_size = 0;
    if (network_type == ArtificialNeuralNetwork::JORDAN_NETWORK) {
        recurrent_layer_size = 1;
    } else if (network_type == ArtificialNeuralNetwork::ELMAN_NETWORK) {
        recurrent_layer_size = flight_columns;
    }

    if (network_type != ArtificialNeuralNetwork::FEED_FORWARD_NETWORK) recurrent_layers = 1;

    get_argument(arguments, "--seconds_into_future", false, seconds_into_future);

    if (rank == 0) {
        cout << "#seconds into future: " << seconds_into_future << endl;

        cout << "#network type: " << network_type << endl;

        cout << "#input  layer:         " << input_layer_size << endl;
        cout << "#hidden layer:         " << hidden_layers << " x " << flight_columns << endl;
        cout << "#output layer:         " << output_str << " : " << output_target << endl;
        cout << "#recurrent layer:      " << recurrent_layers << " x " << recurrent_layer_size << endl;
    }

    srand48(time(NULL));

    bool use_bias = false;
    use_bias = argument_exists(arguments, "--use_bias");

    ann = new ArtificialNeuralNetwork(input_layer_size, hidden_layers, flight_columns, 1, recurrent_layer_size, use_bias, network_type);

    string ann_parameters;
    if (get_argument(arguments, "--test_ann", false, ann_parameters)) {
        cout << "#testing ann: '" << ann_parameters << "'" << endl;
        vector<double> ann_parameters_v;
        string_to_vector(ann_parameters, ann_parameters_v);

        /*
        for (unsigned int i = 0; i < ann_parameters_v.size(); i++) {
            cout << "ann_parameters_v[" << i << "]: " << ann_parameters_v[i] << endl;
        }
        */

        double total_error = 0;
        double current_error;
        for (unsigned int i = 0; i < flight_rows - (input_lags + 1 + output_timesteps + seconds_into_future); i++) {
            current_error = ann->evaluate( ann_parameters_v, &(flight_data[i * flight_columns]), &(flight_data[((i + input_lags + 1 + seconds_into_future) * flight_columns) + output_target]) );

            cout << setw(5) << i;
            for (unsigned int j = 0; j < flight_columns; j++) {
                cout << setw(20) << ann->get_output_layer(j) << setw(20) << flight_data[((i + input_lags + 1 + seconds_into_future) * flight_columns) + j];
            }
            cout << endl;

            total_error += current_error;
        }
        cerr << "total error: " << (sqrt(total_error) / flight_rows) << endl;

    } else {
        int number_of_nodes = 0;
        unsigned int hidden_layer_size = flight_columns;
        unsigned int output_layer_size = 1;

        if (hidden_layers > 0) {
            number_of_nodes = (input_layer_size * hidden_layer_size) +      //weights from input layer to 1st hidden layer
                              (hidden_layer_size * recurrent_layer_size) +  //weights from recurrent layer to 1st hidden layer
                              (hidden_layer_size * (hidden_layers - 1)) +   //weights between hidden layers
                              (hidden_layer_size * output_layer_size);      //weights from last hidden lyaer to output layer

        } else {
            number_of_nodes = (input_layer_size * output_layer_size) +      //weights from input layer to output layer
                              (recurrent_layer_size * output_layer_size);   //can have a jordan network with no hidden layers
        }

        if (use_bias) {
            number_of_nodes += 1 +                             //weights to output_layer
                               (flight_columns * hidden_layers);            //bias weights to each hidden layer
        }

        vector<double> min_bound(number_of_nodes, -1.5);
        vector<double> max_bound(number_of_nodes, 1.5);

        if (rank == 0) {
            cout << "number of parameters: " << min_bound.size() << endl;
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

#ifdef CUDA
            int max_rank;
            MPI_Comm_size(MPI_COMM_WORLD, &max_rank);

            if (max_rank == 33) {
                int device_assignments[] = {-1, 0, 1, -1, -1, -1, -1, -1, -1,
                    0, 1, -1, -1, -1, -1, -1, -1,
                    0, 1, -1, -1, -1, -1, -1, -1,
                    0, 1, -1, -1, -1, -1, -1, -1};

                ps.go(objective_function, objective_function_gpu, device_assignments);
            } else if (max_rank == 9) {
                int device_assignments[] = {-1, 0, 1, 0, 1, 0, 1, 0, 1};

                ps.go(objective_function, objective_funciton_gpu, device_assignments);
            } else if (max_rank == 2) {
                ps.go(objective_function_gpu);
            }
#else
            ps.go(objective_function);
#endif

        } else if (search_type.compare("de_mpi") == 0) {
            DifferentialEvolutionMPI de(min_bound, max_bound, arguments);
            de.go(objective_function);

        } else if (search_type.compare("snm") == 0 || search_type.compare("gd") == 0 || search_type.compare("cgd") == 0) {
            srand48(time(NULL));

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
                synchronous_gradient_descent(arguments, objective_function, starting_point, step_size);
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
