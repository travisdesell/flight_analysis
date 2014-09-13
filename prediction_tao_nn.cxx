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

#include <cmath>

#include "mpi.h"

#include "flight_io.hxx"

//from TAO
#include "neural_networks/edge.hxx"
#include "neural_networks/time_series_neural_network.hxx"

#include "mpi/mpi_particle_swarm.hxx"
#include "mpi/mpi_differential_evolution.hxx"

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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, max_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &max_rank);

    vector<string> arguments(argv, argv + argc);

    string input_filename;
    get_argument(arguments, "--input_filename", true, input_filename);

    //read the flight data
    unsigned int time_series_rows;
    unsigned int time_series_columns;
    double* flight_data = NULL;
    vector<string> column_headers;
    read_flight_file(input_filename, time_series_rows, time_series_columns, flight_data, column_headers);

    cerr << "# time series columns = " << time_series_columns << endl;
    cerr << "# time series rows = " << time_series_rows << endl;

    column_headers.erase(column_headers.begin());
    column_headers.erase(column_headers.begin());
    cerr << "#";
    for (int i = 0; i < column_headers.size(); i++) cerr << " " << column_headers[i];
    cerr << endl;

    //set the time series data from the flight data
    double **time_series_data;

    //normalize the data
    vector<double> mins(time_series_columns, numeric_limits<double>::max());
    vector<double> maxs(time_series_columns, -numeric_limits<double>::max());

    time_series_data = new double*[time_series_rows];
    for (int i = 0; i < time_series_rows; i++) {
        time_series_data[i] = new double[time_series_columns];
        for (int j = 0; j < time_series_columns; j++) {
            time_series_data[i][j] = flight_data[(i * time_series_columns) + j];

            if (time_series_data[i][j] < mins[j]) mins[j] = time_series_data[i][j];
            if (time_series_data[i][j] > maxs[j]) maxs[j] = time_series_data[i][j];
        }
    }

    cerr << "#";
    for (int i = 0; i < time_series_columns; i++) cerr << " " << mins[i];
    cerr << endl;

    cerr << "#";
    for (int i = 0; i < time_series_columns; i++) cerr << " " << maxs[i];
    cerr << endl;

    for (int i = 0; i < time_series_rows; i++) {
        for (int j = 0; j < time_series_columns; j++) {

            /*
            if (0 == column_headers[j].compare("roll_attitude")) {
                time_series_data[i][j] = time_series_data[i][j] / 60;

            } else if (0 == column_headers[j].compare("pitch_attitude")) {
                time_series_data[i][j] = time_series_data[i][j] / 60;

            } else if (0 == column_headers[j].compare("indicated_airspeed")) {
                time_series_data[i][j] = time_series_data[i][j] / 200;

            } else if (0 == column_headers[j].compare("msl_altitude")) {
                time_series_data[i][j] = time_series_data[i][j] / 6000;

            } else {
            */
                time_series_data[i][j] = (time_series_data[i][j] - mins[j]) / (maxs[j] - mins[j]);
            //}
        }
    }


    string output_str;
    get_argument(arguments, "--output_target", true, output_str);

    int output_target = -1;
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

    long seed;
    if (get_argument(arguments, "--seed", false, seed)) {
        srand48(seed);
    } else {
        srand48(time(NULL));
    }

    string nn_filename;
    get_argument(arguments, "--nn", true, nn_filename);

    ts_nn = new TimeSeriesNeuralNetwork(output_target);
    ts_nn->set_time_series_data(time_series_data, time_series_rows, time_series_columns);
    ts_nn->read_nn_from_file(nn_filename);

    string weights_filename;
    if (get_argument(arguments, "--weights", false, weights_filename)) {
        //read the nn edges and weights from a file, then run it once
        ts_nn->reset();
        ts_nn->read_weights_from_file(weights_filename);
        cout << "total error: " << ts_nn->evaluate() << endl;

    } else {
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
