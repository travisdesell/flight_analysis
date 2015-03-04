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

#include <sstream>
using std::ostringstream;

#include <cmath>

//from Boost
#include <boost/filesystem.hpp>
using boost::filesystem::exists;
using boost::filesystem::remove;
using boost::filesystem::copy_file;

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


string get_sort_filename(string dir, string filename) {
    ostringstream oss;
    oss << dir << filename.substr( filename.rfind('/') + 1 );
    return oss.str();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, max_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &max_rank);

    vector<string> arguments(argv, argv + argc);

    //read the flight data

    //make a vector of the parameters we're going to analyze with the NN
    //JIM SUGGESTS THESE:

    /*
    flight_parameters.push_back("ALT_STD");    //ALT_STD Pressure Alt    
    flight_parameters.push_back("ALT_CPT");    //ALT_CPT Baro 1 Altimeter (inHg) 
    flight_parameters.push_back("ALT_FO");     //ALT_FO  Baro 2 Altimeter (inHg) 
    flight_parameters.push_back("ALT_SEL");    //ALT_SEL Selected altitude   
    flight_parameters.push_back("AOAL");
    flight_parameters.push_back("AOAR");
    flight_parameters.push_back("FF1");        //fuel flow - engine 1
    flight_parameters.push_back("FF2");        //fuel flow - engine 2
    flight_parameters.push_back("GS");
    flight_parameters.push_back("IAS");        //indicated airspeed
    flight_parameters.push_back("ITT_1");      //interstage turbine temp - engine 1
    flight_parameters.push_back("ITT_2");      //interstage turbine temp - engine 2
    flight_parameters.push_back("IVV_R");      //vertical speed (feet per minute)
    flight_parameters.push_back("BLD_PRS1");   //bleed pressure (psi) - engine 1
    flight_parameters.push_back("BLD_PRS2");   //bleed pressure (psi) - engine 2
    flight_parameters.push_back("HYD_PRS1");   //hydrolic pressure valve fully closed - engine 1
    flight_parameters.push_back("HYD_PRS2");   //hydrolic pressure valve fully closed - engine 2
    flight_parameters.push_back("N11");
    flight_parameters.push_back("N12");
    flight_parameters.push_back("N21");
    flight_parameters.push_back("N22");
    flight_parameters.push_back("OIL_PRS_L");
    flight_parameters.push_back("OIL_PRS_R");
    flight_parameters.push_back("OIL_QTY1");
    flight_parameters.push_back("OIL_QTY2");
    flight_parameters.push_back("OIL_TMP1");
    flight_parameters.push_back("OIL_TMP2");
    flight_parameters.push_back("PITCH");
    flight_parameters.push_back("PITCH2");
    flight_parameters.push_back("PLA1");
    flight_parameters.push_back("PLA2");
    flight_parameters.push_back("ROLL");
    flight_parameters.push_back("ROLL_TRIM_P");
    flight_parameters.push_back("RUDD");
    flight_parameters.push_back("RUDD_TRIM_P");
    flight_parameters.push_back("SAT");
    flight_parameters.push_back("TAT");
    flight_parameters.push_back("VIB_N11");
    flight_parameters.push_back("VIB_N12");
    flight_parameters.push_back("VIB_N21");
    flight_parameters.push_back("VIB_N22");
    */

    bool is_endeavor;
    get_argument(arguments, "--is_endeavor", true, is_endeavor);

    vector<string> flight_files;
    get_argument_vector(arguments, "--flight_files", true, flight_files);

    vector<string> flight_parameters;
    get_argument_vector(arguments, "--use_parameters", true, flight_parameters);

    string output_directory;
    get_argument(arguments, "--output_directory", true, output_directory);

    int n_flights = 0;
    vector<uint32_t> rows, columns;
    double ***flight_data = NULL;
    read_flights(flight_files, flight_parameters, n_flights, rows, columns, flight_data, is_endeavor);

    cerr << "#";
    for (int i = 0; i < flight_parameters.size(); i++) cerr << " " << flight_parameters[i];
    cerr << endl;

    for (int i = 0; i < n_flights; i++) {
//        normalize_data(flight_data[i], rows[i], columns[i]);

        ostringstream oss;
        oss << output_directory << flight_files[i].substr(flight_files[i].rfind("/")) << endl;

        cout << "writing to: '" << oss.str() << "'";

        write_flight_data(oss.str(), flight_parameters, flight_data[i], rows[i], columns[i]);
    }

    return 0;
}
