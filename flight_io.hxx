#ifndef FLIGHT_IO_H
#define FLIGHT_IO_H

#include <string>
using std::string;

#include <vector>
using std::vector;


void get_flight_files_from_directory(string directory, vector<string> &flight_files);
void get_flight_files_from_file(string list_filename, vector<string> &flight_files);

void read_flights(const vector<string> &input_files, const vector<string> &column_headers, int &n_files, vector<unsigned int> &rows, vector<unsigned int> &cols, double *** &final_flight_data, bool endeavor_data);

void read_flight_file(string input_filename, const vector<string> &column_headers, unsigned int &rows, unsigned int &cols, double** &final_flight_data, bool endeavor_data);

void normalize_data_sets(const vector<string> &column_headers, int n_flights1, double ***data1, const vector<unsigned int> &rows1, const vector<unsigned int> &col1, int n_flights2, double ***data2, const vector<unsigned int> &rows2, const vector<unsigned int> &col2);

void normalize_data(double **data, int rows, int columns);

void get_output_data(double **input_data, int rows, int cols, const vector<string> &input_headers, const vector<string> &output_headers, double ***output_data);

void write_flight_data(string output_file, const vector<string> &column_names, double **data, int rows, int columns);

#endif
