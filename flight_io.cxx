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
    ifstream input_file( input_filename );
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
    cols = column_headers.size();
    cols = 3;

    vector< vector<double> > flight_data;
    getline( input_file, s );
    while (input_file.good()) {
//        std::cout << s << "\n";

        vector<double> flight_row( column_headers.size() );

        int position = 0;

        char_separator<char> sep(" ", "");
        tokenizer<char_separator<char> > tok(s, sep);

        for (tokenizer< char_separator<char> >::iterator i = tok.begin(); i != tok.end(); ++i) {
            if (position >= 3) continue;

            flight_row[position] = atof( (*i).c_str() );
//            cout << "pushed back '" << *i << "'" << endl;
            position++;
        }

        flight_data.push_back(flight_row);

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
