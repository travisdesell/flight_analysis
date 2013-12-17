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

#include <string>
using std::string;

#include <vector>
using std::vector;

#include <boost/lexical_cast.hpp>
using boost::lexical_cast;

#include <boost/tokenizer.hpp>
using boost::tokenizer;
using boost::char_separator;


void read_flight_file(string input_filename, unsigned int &rows, unsigned int &cols, double* &final_flight_data, vector<string> &column_headers) {
    ifstream input_file( input_filename.c_str() );
    if (!input_file.is_open()) {
        cerr << "Error, could not open file: '" << input_filename << "' for reading." << endl;
        exit(1);
    }

    string s;
    getline( input_file, s );

    char_separator<char> sep(" ", "");
    tokenizer<char_separator<char> > tok(s, sep);
    for (tokenizer< char_separator<char> >::iterator i = tok.begin(); i != tok.end(); ++i) {
        column_headers.push_back( *i );
//        cout << "pushed back '" << *i << "'" << endl;
    }
    cols = column_headers.size() - 2;
    cout << "cols: " << cols << endl;

    vector< vector<double> > flight_data;
    getline( input_file, s );
    while (input_file.good()) {
        if (s.length() > 0 && s[0] == '#') {
            getline( input_file, s);
            continue;
        }
//        std::cout << s << "\n";

        vector<double> flight_row( cols );

        int position = 0;

        char_separator<char> sep(" ", "");
        tokenizer<char_separator<char> > tok(s, sep);

        for (tokenizer< char_separator<char> >::iterator i = tok.begin(); i != tok.end(); ++i) {
            if (position == 0) {
                position++;
                continue;
            }
//            cout << "token is: '" << *i << "', assigning to position: " << (position - 1) << endl;

            flight_row.at(position - 1) = atof( (*i).c_str() );
            position++;
        }

        /*
        cout << "pushing back" << endl;
        cout << "flight row:" << endl;
        for (int i = 0; i < flight_row.size(); i++) {
            cout << " " << flight_row[i];
        }
        cout << endl;
        cout << "flight_data.size(): " << flight_data.size() << ", capacity: " << flight_data.capacity() << endl;
        */
//        if (flight_row[2] > 0) {    //only use rows with indicated airspeed > 0
            flight_data.push_back(flight_row);
//        }

//        cout << "getting next line" << endl;
        getline( input_file, s);
    }
    rows = flight_data.size();

    final_flight_data = new double[rows * cols];
    for (unsigned int i = 0; i < rows; i++) {
        for (unsigned int j = 0; j < cols; j++) {
            final_flight_data[(i * cols) + j] = flight_data[i][j];
        }
    }
}
