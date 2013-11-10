#ifndef FLIGHT_IO_H
#define FLIGHT_IO_H

#include <string>
using std::string;

#include <vector>
using std::vector;

void read_flight_file(string input_filename, unsigned int &rows, unsigned int &cols, double* &final_flight_data, vector<string> &column_headers);

#endif
