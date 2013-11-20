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

#include "synchronous_algorithms/parameter_sweep.hxx"

//from undvc_common
#include "undvc_common/arguments.hxx"

#define PRECISION double

class ArtificialNeuralNetwork {
    public:
        virtual PRECISION evaluate(const vector<PRECISION> &weights, const double *input_layer, const double *expected_output) = 0;
};

class RecurrentANN : public ArtificialNeuralNetwork {
    private:
        const unsigned int input_layer_size;
        const unsigned int hidden_layer_size;
        const unsigned int output_layer_size;
        const unsigned int recurrent_layer_size;

        PRECISION *hidden_layer;
        PRECISION *output_layer;
        PRECISION *recurrent_layer;

    public:
        RecurrentANN(unsigned int ils, unsigned int hls, unsigned int ols) : input_layer_size(ils), hidden_layer_size(hls), output_layer_size(ols), recurrent_layer_size(ols) {
            hidden_layer = new PRECISION[hidden_layer_size];
            output_layer = new PRECISION[output_layer_size];

            recurrent_layer = new PRECISION[recurrent_layer_size];
            for (unsigned int i = 0; i < recurrent_layer_size; i++) recurrent_layer[i] = 0;
        }

        PRECISION evaluate(const vector<PRECISION> &weights, const double *input_layer, const double *expected_output) {
            unsigned int current_weight = 0;
            for (unsigned int i = 0; i < hidden_layer_size; i++) {
                hidden_layer[i] = 0;

                for (unsigned int j = 0; j < input_layer_size; j++) {
                    hidden_layer[i] += weights[current_weight] * input_layer[j];
                    current_weight++;
                }
            }

            for (unsigned int i = 0; i < hidden_layer_size; i++) {
                for (unsigned int j = 0; j < recurrent_layer_size; j++) {
                    hidden_layer[i] += weights[current_weight] * recurrent_layer[j];
                    current_weight++;
                }
            }

            for (unsigned int i = 0; i < output_layer_size; i++) {
                output_layer[i] = 0;

                for (unsigned int j = 0; j < hidden_layer_size; j++) {
                    output_layer[i] += weights[current_weight] * hidden_layer[j];
                    current_weight++;
                }
            }

            PRECISION error = 0.0;
            PRECISION temp;
            for (unsigned int i = 0; i < output_layer_size; i++) {
                temp = output_layer[i] - expected_output[i];
                error += temp * temp;
            }

            return sqrt(error);
        }

//    friend double objective_function(const vector<double> &);
};


class FeedForwardANN : public ArtificialNeuralNetwork{
    private:
        const unsigned int input_layer_size;
        const unsigned int hidden_layer_size;
        const unsigned int output_layer_size;

        PRECISION *hidden_layer;
        PRECISION *output_layer;

    public:
        FeedForwardANN(unsigned int ils, unsigned int hls, unsigned int ols) : input_layer_size(ils), hidden_layer_size(hls), output_layer_size(ols) {
            hidden_layer = new PRECISION[hidden_layer_size];
            output_layer = new PRECISION[output_layer_size];
        }

        PRECISION evaluate(const vector<PRECISION> &weights, const double *input_layer, const double *expected_output) {
            unsigned int current_weight = 0;
            for (unsigned int i = 0; i < hidden_layer_size; i++) {
                hidden_layer[i] = 0;

                for (unsigned int j = 0; j < input_layer_size; j++) {
                    hidden_layer[i] += weights[current_weight] * input_layer[j];
                    current_weight++;
                }
            }

            for (unsigned int i = 0; i < output_layer_size; i++) {
                output_layer[i] = 0;

                for (unsigned int j = 0; j < hidden_layer_size; j++) {
                    output_layer[i] += weights[current_weight] * hidden_layer[j];
                    current_weight++;
                }
            }

            PRECISION error = 0.0;
            PRECISION temp;
            for (unsigned int i = 0; i < output_layer_size; i++) {
                temp = output_layer[i] - expected_output[i];
                error += temp * temp;
            }

            return sqrt(error);
        }

//    friend double objective_function(const vector<double> &);
};

double* flight_data = NULL;
unsigned int flight_rows;
unsigned int flight_columns;
int input_timesteps;
int output_timesteps;
ArtificialNeuralNetwork *ann;

double objective_function(const vector<double> &parameters) {

//    double max_error = 0;
    double total_error = 0;
    double current_error;
    for (int i = 0; i < flight_rows - (input_timesteps + output_timesteps); i++) {
        current_error = ann->evaluate( parameters, &(flight_data[i * flight_columns]), &(flight_data[(i + input_timesteps) * flight_columns]) );
//        if (current_error > max_error) max_error = current_error;

        total_error += current_error;
//        cout << setw(15) << total_error << " - " << setw(10) << current_error << " - " << max_error << endl;
    }
    return -(total_error / flight_rows);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    vector<string> arguments(argv, argv + argc);
    
    vector<string> column_headers;

    string input_filename;
    get_argument(arguments, "--input_filename", true, input_filename);
    read_flight_file(input_filename, flight_rows, flight_columns, flight_data, column_headers);

    /*
    for (int i = 0; i < flight_rows; i++) {
        for (int j = 0; j < flight_columns; j++) {
            cout << setw(10) << flight_data[(i * flight_columns) + j];
        }
        cout << endl;
    }
    */

    //determine how many previous timesteps will be fed into the neural network
    get_argument(arguments, "--input_timesteps", true, input_timesteps);
    get_argument(arguments, "--output_timesteps", true, output_timesteps);

    unsigned int input_layer_size = input_timesteps * flight_columns;
    unsigned int output_layer_size = output_timesteps * flight_columns;

//    int hidden_layer_size  = (input_layer_size + output_layer_size) * 0.2;
    unsigned int hidden_layer_size;
    get_argument(arguments, "--hidden_nodes", true, hidden_layer_size);

    unsigned int recurrent_layer_size = 0;
    if (argument_exists(arguments, "--recurrent")) {
        recurrent_layer_size = output_layer_size;
    }

    cout << "input  timesteps: " << input_timesteps << endl;
    cout << "output timesteps: " << output_timesteps << endl;

    cout << "input     layer size: " << input_layer_size << endl;
    cout << "hidden    layer size: " << hidden_layer_size << endl;
    cout << "output    layer size: " << output_layer_size << endl;
    cout << "recurrent layer size: " << recurrent_layer_size << endl;

    srand48(time(NULL));

    if (recurrent_layer_size > 0) {
        ann = new RecurrentANN(input_layer_size, hidden_layer_size, output_layer_size);
    } else {
        ann = new FeedForwardANN(input_layer_size, hidden_layer_size, output_layer_size);
    }

    vector<double> min_bound((input_layer_size * hidden_layer_size) + (hidden_layer_size * output_layer_size) + (hidden_layer_size * recurrent_layer_size), -20.0);
    vector<double> max_bound((input_layer_size * hidden_layer_size) + (hidden_layer_size * output_layer_size) + (hidden_layer_size * recurrent_layer_size), 20.0);

    cout << "number of parameters: " << min_bound.size() << endl;

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

    } else {
        fprintf(stderr, "Improperly specified search type: '%s'\n", search_type.c_str());
        fprintf(stderr, "Possibilities are:\n");
        fprintf(stderr, "    de     -       differential evolution\n");
        fprintf(stderr, "    ps     -       particle swarm optimization\n");
        exit(0);
    }

}
