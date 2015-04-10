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

    string flights_dir;
    get_argument(arguments, "--flights_dir", true, flights_dir);

    vector<string> flight_files;
    get_flight_files_from_directory(flights_dir, flight_files);

    int n_flights = 0;
    vector<uint32_t> rows, columns;
    double ***flight_data = NULL;
    read_flights(flight_files, column_headers, n_flights, rows, columns, flight_data, true);

    cerr << "#";
    for (int i = 0; i < column_headers.size(); i++) cerr << " " << column_headers[i];
    cerr << endl;

    int class2_count = 0;
    int class3_count = 0;

    int vib_n11_pos, vib_n12_pos;
    for (int i = 0; i < column_headers.size(); i++) {
        if (column_headers[i].compare("VIB_N11") == 0) vib_n11_pos = i;
        if (column_headers[i].compare("VIB_N12") == 0) vib_n12_pos = i;
    }

    cout << "VIB_N11 at " << vib_n11_pos << ", VIB_N12 at " << vib_n12_pos << endl;

    string no_event_dir = "/Volumes/Macintosh HD 2/endeavor_data/vib_event_none/";
    string class2_dir = "/Volumes/Macintosh HD 2/endeavor_data/vib_event_class2/";
    string class3_dir = "/Volumes/Macintosh HD 2/endeavor_data/vib_event_class3/";

    for (int i = 0; i < n_flights; i++) {
        double max_vib11 = 0, max_vib12 = 0;

        for (int j = 0; j < rows[i]; j++) {
            if (flight_data[i][j][vib_n11_pos] > max_vib11) max_vib11 = flight_data[i][j][vib_n11_pos];
            if (flight_data[i][j][vib_n12_pos] > max_vib12) max_vib12 = flight_data[i][j][vib_n12_pos];

            if (max_vib11 > 1.75 || max_vib12 > 1.75) break;
        }

        cout << "flight[" << i << "] max vib11: " << max_vib11 << ", max vib12: " << max_vib12 << endl;

        string sort_filename;
        if (max_vib11 > 1.75 || max_vib12 > 1.75) {
            cout << "class 3 event!" << endl;
            class3_count++;
            sort_filename = get_sort_filename(class3_dir, flight_files[i]);

        } else if (max_vib11 > 1.5 || max_vib12 > 1.5) {
            cout << "class 2 event!" << endl;
            class2_count++;
            sort_filename = get_sort_filename(class2_dir, flight_files[i]);

        } else {
            cout << "no event!" << endl;
            sort_filename = get_sort_filename(no_event_dir, flight_files[i]);
        }

        if (exists(sort_filename)) {
            cout << "file already exists!" << endl;
            //remove (to_fp);
        } else {
            cout << "copying from: '" << flight_files[i] << " to '" << sort_filename << "'" << endl;
            copy_file(flight_files[i], sort_filename);
        }
    }


    return 0;
}
