#include <cmath>

#include <fstream>
using std::ofstream;
using std::ifstream;

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <iomanip>
using std::setw;

#include <map>
using std::map;

#include <limits>
using std::numeric_limits;

#include <sstream>
using std::ostringstream;

#include <string>
using std::string;

#include <vector>
using std::vector;


#include <boost/lexical_cast.hpp>
using boost::lexical_cast;

#include <boost/filesystem.hpp>
using boost::filesystem::exists;
using boost::filesystem::directory_iterator;

#include <boost/tokenizer.hpp>
using boost::tokenizer;
using boost::char_separator;

//For MYSQL
#include "mysql.h"

//For argument parsing and other helpful functions
#include "arguments.hxx"

template <typename T>
string to_string(T value) {
    ostringstream oss;
    oss << value;
    return oss.str();
}

/**
 * Columns are: id, flight, phase, time, msl_altitude, indicated_airspeed, vertical_airspeed, tas, heading, course, pitch_attitude, roll_attitude, eng_1_rpm, vertical_acceleration, longitudinal_acceleration, lateral_acceleration, oat, groundspeed, latitude, longitude, nav_1_freq, nav_2_freq, obs_1, fuel_quantity_left_main, fuel_quantity_right_main, eng_1_fuel_flow, eng_1_oil_press, eng_1_oil_temp, eng_1_cht_1, eng_1_cht_2, eng_1_cht_3, eng_1_cht_4, eng_1_egt_1, eng_1_egt_2, eng_1_egt_3, eng_1_egt_4, system_1_volts, system_2_volts, system_1_amps, system_2_amps
 */

int main(int argc /* number of command line arguments */, char **argv /* command line argumens */) {
    vector<string> arguments(argv, argv + argc);

    string input_directory = "./";
    get_argument(arguments, "--input_directory", false, input_directory);

    cout << "using '" << input_directory << "' as the input directory, this can be changed with the --input_directory command line argument" << endl;

    if ( !exists( input_directory ) ) return false;

    string output_filename;
    get_argument(arguments, "--output_file", true, output_filename);
    ofstream output_file( (input_directory + output_filename).c_str() );

    directory_iterator end_iterator; // default construction yields past-the-end
    for (directory_iterator iterator( input_directory ); iterator != end_iterator; ++iterator) {

        if ( is_directory(iterator->status()) ) {
            cout << "processing files for: " << iterator->path() << endl;

            for(directory_iterator sub_iterator( iterator->path() ); sub_iterator != end_iterator; ++sub_iterator) {
                cout << "file:     " << sub_iterator->path().c_str() << endl;

                ifstream input_file( sub_iterator->path().c_str() );
                if (!input_file.is_open()) {
                    cerr << "Error, could not open file: '" << sub_iterator->path().filename().c_str() << "' for reading." << endl;
                    exit(1);
                }

                vector< vector<string>* > flight_data;

                while (input_file.good()) {
                    string s;
                    getline( input_file, s );

                    if (!input_file.good()) break;
//                    std::cout << s << "\n";

                    vector<string> *flight_row = new vector<string>();

                    char_separator<char> sep(" ", "");
                    tokenizer<char_separator<char> > tok(s, sep);
                    for (tokenizer< char_separator<char> >::iterator i = tok.begin(); i != tok.end(); ++i) {
                        flight_row->push_back( *i );
//                        cout << "pushed back '" << *i << "'" << endl;
                    }

                    flight_data.push_back(flight_row);
                }

                int number_rows = flight_data.size();
                int number_columns = flight_data.at(0)->size();

                double **fd = new double*[number_rows];
                for (int i = 0; i < number_rows; i++) {
                    fd[i] = new double[number_columns];
                    for (int j = 0; j < number_columns; j++) {
                        fd[i][j] = fabs( lexical_cast<double>( flight_data.at(i)->at(j) ) );
                    }
                }

                while ( !flight_data.empty() ) {
                    vector<string> *flight_row = flight_data.back();
                    flight_data.pop_back();
                    delete flight_row;
                }

                //Need to calculate the minimum, maximum, average for each column.
                //Also, minimum, maximum, average change
                //And minimum, maximum, average change of change (over a given window)

                double *values = new double[ number_columns * 9 ];
                for (int i = 0; i < number_columns; i++) {
                    values[i * 9 + 0] = numeric_limits<double>::max();  //min
                    values[i * 9 + 1] = 0.0;                          //avg
                    values[i * 9 + 2] = -numeric_limits<double>::max();  //max

                    values[i * 9 + 3] = numeric_limits<double>::max();  //roc min
                    values[i * 9 + 4] = 0.0;                          //roc avg
                    values[i * 9 + 5] = -numeric_limits<double>::max();  //roc max

                    values[i * 9 + 6] = numeric_limits<double>::max();  //roc^2 min
                    values[i * 9 + 7] = 0.0;                          //roc^2 avg
                    values[i * 9 + 8] = -numeric_limits<double>::max();  //roc^2 max

                    for (int j = 0; j < number_rows; j++) {
                        if (values[i * 9 + 0] > fd[j][i]) values[i * 9 + 0] = fd[j][i];
                        values[i * 9 + 1] += fd[j][i];
                        if (values[i * 9 + 2] < fd[j][i]) values[i * 9 + 2] = fd[j][i];

                        if (j >= 1) {
                            double rate_of_change = fd[j][i] - fd[j-1][i];

                            if (values[i * 9 + 3] > rate_of_change) values[i * 9 + 3] = rate_of_change;
                            values[i * 9 + 4] += rate_of_change;
                            if (values[i * 9 + 5] < rate_of_change) values[i * 9 + 5] = rate_of_change;

                            if (j >= 2) {
                                double rate_of_rate_of_change = rate_of_change - (fd[j-1][i] - fd[j-2][i]);

                                if (values[i * 9 + 6] > rate_of_rate_of_change) values[i * 9 + 6] = rate_of_rate_of_change;
                                values[i * 9 + 7] += rate_of_rate_of_change;
                                if (values[i * 9 + 8] < rate_of_rate_of_change) values[i * 9 + 8] = rate_of_rate_of_change;
                            }
                        }
                    }

                    values[i * 9 + 1] /= number_rows;
                    values[i * 9 + 4] /= number_rows - 1;
                    values[i * 9 + 7] /= number_rows - 2;
                }

                cout << sub_iterator->path().leaf().c_str() << " ";
                output_file << sub_iterator->path().leaf().c_str()  << " ";

                cout << iterator->path().leaf().c_str()  << endl;
                output_file << iterator->path().leaf().c_str()  << " ";

                if (iterator->path().leaf().filename() == "no_excedence") {
                    output_file << "0"; //flag the file as not having any excedences
                } else {
                    output_file << "1"; //flag the file as having excedences
                }

                for (int i = 0; i < number_columns; i++) {
                    for (int j = 0; j < 9; j++) {
                        cout << setw(15) << values[i * 9 + j];
                        output_file << " " << values[i * 9 + j];
                    }
                    cout << endl;
                }
                cout << endl;
                output_file << endl;
            }

        } else {
            cout << "skipping file: " << iterator->path() << endl;
        }
    }

}
