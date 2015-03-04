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
using std::setprecision;

#include <limits>
using std::numeric_limits;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include <boost/filesystem.hpp>
using boost::filesystem::directory_iterator;
using boost::filesystem::is_directory;

#include <boost/lexical_cast.hpp>
using boost::lexical_cast;

#include <boost/tokenizer.hpp>
using boost::tokenizer;
using boost::char_separator;

#include "flight_io.hxx"

void get_flight_files_from_file(string list_filename, vector<string> &flight_files) {
    ifstream input_file( list_filename.c_str() );
    if (!input_file.is_open()) {
        cerr << "Error, could not open file: '" << list_filename << "' for reading." << endl;
        exit(1);
    }

    string line;
    getline( input_file, line);
    while (input_file.good()) {
        flight_files.push_back(line);

        getline(input_file, line);
    }
    input_file.close();
}

void get_flight_files_from_directory(string directory, vector<string> &flight_files) {
    directory_iterator end_itr;
    for (directory_iterator itr(directory); itr != end_itr; itr++) {
        if (!is_directory(itr->status())) {
            //skip non-csv files
            if (itr->path().leaf().string().find(".csv") == std::string::npos) {
                cout << "skipping: " << itr->path().c_str() << endl;
                continue;
            }

            //skip PQ files because they have different parameters
            if (itr->path().leaf().string().find("PQ") != std::string::npos) {
                cout << "skipping: " << itr->path().c_str() << endl;
                continue;
            }

            if (itr->path().leaf().c_str()[0] == '.') {
                cout << "skipping: " << itr->path().c_str() << endl;
                continue;
            }   

            cout << "adding input file: '" << itr->path().c_str() << endl;

            flight_files.push_back(itr->path().c_str());
        }
    }   
    cout << "parsed " << flight_files.size() << " files from '" << directory << "'" << endl;
}

void read_flights(const vector<string> &input_files, const vector<string> &column_headers, int &n_files, vector<unsigned int> &rows, vector<unsigned int> &cols, double *** &final_flight_data, bool endeavor_data) {
    n_files = input_files.size();
    rows = vector<unsigned int>(n_files, 0);
    cols = vector<unsigned int>(n_files, 0);

    final_flight_data = new double**[n_files];
    for (uint32_t i = 0; i < input_files.size(); i++) {
        cout << "reading from file: '" << input_files.at(i) << "'" << endl;
        read_flight_file(input_files.at(i), column_headers, rows.at(i), cols.at(i), final_flight_data[i], endeavor_data);
        cout << "file " << i << " of " << input_files.size() << ", read rows: " << rows.at(i) << " and cols: " << cols.at(i) << endl;

        /*
        for (int j = 0; j < rows.at(i); j++) {
            for (int k = 0; k < cols.at(i); k++) {
                cout << setw(8) << final_flight_data[i][j][k];
            }
            cout << endl;
        }
        */
    }
}


void read_flight_file(string input_filename, const vector<string> &column_headers, unsigned int &rows, unsigned int &cols, double** &final_flight_data, bool endeavor_data) {
    ifstream input_file( input_filename.c_str() );
    if (!input_file.is_open()) {
        cerr << "Error, could not open file: '" << input_filename << "' for reading." << endl;
        exit(1);
    }

    string s;
    getline( input_file, s );
    while (s.size() > 0 && s[0] == '#') getline(input_file, s); //skip comments
    if (!input_file.good()) {
        cerr << "Error reading input file: '" << input_filename << "', had problem parsing initial comments." << endl;
        cerr << "comment lines should all be at the top of the file, and begin with the character '#'." << endl;
        exit(1);
    }

    s.erase(std::remove(s.begin(), s.end(), '\r'), s.end());

    vector<string> file_column_headers;
    vector<int> requested_columns(column_headers.size(), -1);

    char_separator<char> sep(" ,\n", "");
    tokenizer<char_separator<char> > tok(s, sep);

    int pos = 0;
    for (tokenizer< char_separator<char> >::iterator i = tok.begin(); i != tok.end(); ++i) {
        file_column_headers.push_back( *i );

        //save the columns that are actually going to be used
        for (int j = 0; j < column_headers.size(); j++) {
            //cout << "comparing " << column_headers[j] << " to " << file_column_headers.back() << endl;
            if (column_headers[j].compare( file_column_headers.back() ) == 0) {
                requested_columns[j] = pos;
                break;
            }
        }
        pos++;
//        cout << "pushed back '" << *i << "'" << endl;
    }

    cols = file_column_headers.size();

    /*
    cout << "requested headers (" << column_headers.size() << "): ";
    for (int i = 0; i < column_headers.size(); i++) {
        cout << " " << column_headers[i];
    }
    cout << endl;

    cout << "header positions  (" << requested_columns.size() << "): ";
    for (int i = 0; i < requested_columns.size(); i++) {
        cout << " " << requested_columns[i];
    }
    cout << endl;
    */

    for (int i = 0; i < requested_columns.size(); i++) {
        for (int j = i + 1; j < requested_columns.size(); j++) {
            if (requested_columns[i] == requested_columns[j]) {
                cerr << "ERROR: column headers matched to duplicate row: " << endl;
                cerr << "column_headers[" << i << "] '" << column_headers[i] << "' matched to column_headers[" << j << "] '" << column_headers[j] << "'" << endl;
                exit(1);
            }
        }
    }

    if (column_headers.size() != requested_columns.size()) {
        cerr << "ERROR: not all column headers requested were found." << endl;
        cerr << "MISSING: " << endl;
        for (int i = 0; i < requested_columns.size(); i++) {
            if (requested_columns[i] < 0) cerr << "    " << column_headers[i] << endl;
        }

        exit(1);
    }

    vector< vector<double> > flight_data;
    if (!endeavor_data) {
        getline( input_file, s );
        while (input_file.good()) {
            if (s.length() > 0 && s[0] == '#') {
                getline( input_file, s);
                continue;
            }
    //        std::cout << s << "\n";

            vector<double> flight_row( cols );

            int position = 0;

            char_separator<char> sep(" ,", "");
            tokenizer<char_separator<char> > tok(s, sep);

            for (tokenizer< char_separator<char> >::iterator i = tok.begin(); i != tok.end(); ++i) {
    //            cout << "token is: '" << *i << "', assigning to position: " << (position - 1) << endl;

                flight_row.at(position) = atof( (*i).c_str() );
                position++;
            }

            /*
            cout << "pushing back flight row:" << endl;
            for (int i = 0; i < flight_row.size(); i++) {
                cout << " '" << flight_row[i] << "'";
            }
            cout << endl;
            cout << "flight_data.size(): " << flight_data.size() << ", capacity: " << flight_data.capacity() << endl;
            */

            flight_data.push_back(flight_row);

            //        cout << "getting next line" << endl;
            getline( input_file, s);
        }
    } else {
        string s1, s2, s3, s4;
        //skip the first three lines, the data is bad
        getline( input_file, s1 );
        getline( input_file, s2 );
        getline( input_file, s3 );

        getline( input_file, s1 );
        getline( input_file, s2 );
        getline( input_file, s3 );
        getline( input_file, s4 );
        s1.erase(std::remove(s1.begin(), s1.end(), '\r'), s1.end());
        s2.erase(std::remove(s2.begin(), s2.end(), '\r'), s2.end());
        s3.erase(std::remove(s3.begin(), s3.end(), '\r'), s3.end());
        s4.erase(std::remove(s4.begin(), s4.end(), '\r'), s4.end());
        s1.erase(std::remove(s1.begin(), s1.end(), ' '), s1.end());
        s2.erase(std::remove(s2.begin(), s2.end(), ' '), s2.end());
        s3.erase(std::remove(s3.begin(), s3.end(), ' '), s3.end());
        s4.erase(std::remove(s4.begin(), s4.end(), ' '), s4.end());

        while (input_file.good()) {
            vector<double> flight_row( cols );

            int position = 0;

            char_separator<char> sep(",", "", boost::keep_empty_tokens);
            tokenizer<char_separator<char> > tok1(s1, sep);
            tokenizer<char_separator<char> > tok2(s2, sep);
            tokenizer<char_separator<char> > tok3(s3, sep);
            tokenizer<char_separator<char> > tok4(s4, sep);

            tokenizer< char_separator<char> >::iterator i1 = tok1.begin();
            tokenizer< char_separator<char> >::iterator i2 = tok2.begin();
            tokenizer< char_separator<char> >::iterator i3 = tok3.begin();
            tokenizer< char_separator<char> >::iterator i4 = tok4.begin();

            while (i1 != tok1.end() && i2 != tok2.end() && i3 != tok3.end() && i4 != tok4.end()) {

                int count = 0;
                flight_row.at(position) = 0;
                if ((*i1).compare("") != 0) {
                    flight_row.at(position) += atof( (*i1).c_str() );
                    count++;
                }

                if ((*i2).compare("") != 0) {
                    flight_row.at(position) += atof( (*i2).c_str() );
                    count++;
                }

                if ((*i3).compare("") != 0) {
                    flight_row.at(position) += atof( (*i3).c_str() );
                    count++;
                }

                if ((*i4).compare("") != 0) {
                    flight_row.at(position) += atof( (*i4).c_str() );
                    count++;
                }

                /*
                if (count == 0) {
                    cerr << "ERROR: parameter '" << file_column_headers[position] << "' not defined for set of 4 lines." << endl;
                    exit(1);
                }
                */

                flight_row.at(position) /= count;
//                cout << "token1 is: '" << *i1 << "', token2 is: '" << *i2 << "', token3 is: '" << *i3 << "', assigning '" << flight_row.at(position) << "' to position: " << (position) << endl;

                position++;
                i1++;
                i2++;
                i3++;
                i4++;
            }

            if (i1 != tok1.end() || i2 != tok2.end() || i3 != tok3.end() || i4 != tok4.end()) {
                cerr << "ERROR: problem reading endeavor data, the three input lines have different numbers of tokens" << endl;
                exit(1);
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
            getline( input_file, s1);
            getline( input_file, s2);
            getline( input_file, s3);
            getline( input_file, s4);
            s1.erase(std::remove(s1.begin(), s1.end(), '\r'), s1.end());
            s2.erase(std::remove(s2.begin(), s2.end(), '\r'), s2.end());
            s3.erase(std::remove(s3.begin(), s3.end(), '\r'), s3.end());
            s4.erase(std::remove(s4.begin(), s4.end(), '\r'), s4.end());
            s1.erase(std::remove(s1.begin(), s1.end(), ' '), s1.end());
            s2.erase(std::remove(s2.begin(), s2.end(), ' '), s2.end());
            s3.erase(std::remove(s3.begin(), s3.end(), ' '), s3.end());
            s4.erase(std::remove(s4.begin(), s4.end(), ' '), s4.end());
        }
    }

    rows = flight_data.size();
    input_file.close();

    int vib_n11_col, vib_n12_col;
    for (int i = 0; i < column_headers.size(); i++) {
        if (column_headers.at(i).compare("VIB_N11") == 0) vib_n11_col = i;
        if (column_headers.at(i).compare("VIB_N12") == 0) vib_n12_col = i;
    }

    final_flight_data = new double*[rows];
    for (unsigned int i = 0; i < rows; i++) {
        final_flight_data[i] = new double[column_headers.size()];
        for (unsigned int j = 0; j < column_headers.size(); j++) {
//            cout << "assigning final_flight_data[" << i << "][" << j << "] = flight_data[" << i << "][" << requested_columns[j] << "]" << endl;
            final_flight_data[i][j] = flight_data[i][requested_columns[j]];
        }

        /*
        for (unsigned int j = 0; j < column_headers.size(); j++) {
            cout << " " << final_flight_data[i][j];
        }
        cout << endl;
        */
    }

    cols = column_headers.size();
}

void normalize_data_sets(const vector<string> &column_headers, int n_flights1, double ***data1, const vector <unsigned int> &rows1, const vector<unsigned int> &cols1, int n_flights2, double ***data2, const vector<unsigned int> &rows2, const vector<unsigned int> &cols2) {

    unsigned int columns = cols1[0];
    for (int i = 1; i < cols1.size(); i++) {
        if (cols1[i] != columns) {
            cerr << "ERROR: different number of columns, read from positives file " << i << endl;
            exit(1);
        }
    }

    for (int i = 0; i < cols2.size(); i++) {
        if (cols2[i] != columns) {
            cerr << "ERROR: different number of columns, read from negatives file " << i << endl;
            exit(1);
        }
    }

    vector<double> all_mins(columns, numeric_limits<double>::max());
    vector<double> all_maxs(columns, -numeric_limits<double>::max());

    vector<double> positive_avgs(columns, 0);
    vector<double> positive_roc(columns, 0);
    vector<double> positive_flight_mins(columns, 0);
    vector<double> positive_flight_maxs(columns, 0);
    vector<double> positive_minavg(columns, 0);
    vector<double> positive_maxavg(columns, 0);

    vector<double> negative_avgs(columns, 0);
    vector<double> negative_roc(columns, 0);
    vector<double> negative_flight_mins(columns, 0);
    vector<double> negative_flight_maxs(columns, 0);
    vector<double> negative_minavg(columns, 0);
    vector<double> negative_maxavg(columns, 0);

    int positive_count = 0;
    cout << "calculating positive min max avg" << endl;
    for (int i = 0; i < n_flights1; i++) {
//        cout << "j from 0 to " << rows1[i] << endl;
        positive_flight_mins.assign(columns, numeric_limits<double>::max());
        positive_flight_maxs.assign(columns, -numeric_limits<double>::max());

        for (int j = 0; j < rows1[i]; j++) {
            for (int k = 0; k < columns; k++) {
                //cout << "checking data " << i << " " << j << " " << k << endl;

                if (data1[i][j][k] < all_mins.at(k)) all_mins.at(k) = data1[i][j][k];
                if (data1[i][j][k] > all_maxs.at(k)) all_maxs.at(k) = data1[i][j][k];
                positive_avgs.at(k) += data1[i][j][k];
                
                //cout << "calcing roc" << endl;
                if (j < rows1[i]-1) positive_roc.at(k) += fabs(data1[i][j][k] - data1[i][j+1][k]);
                //cout << "positive_roc: " << positive_roc.at(k) << endl;

                //cout << "calcing flight mins/maxs" << endl;
                if (data1[i][j][k] < positive_flight_mins.at(k)) positive_flight_mins.at(k) = data1[i][j][k];
                if (data1[i][j][k] > positive_flight_maxs.at(k)) positive_flight_maxs.at(k) = data1[i][j][k];

                //cout << "incrementing pos count" << endl;
                positive_count++;
            }
        }

        for (int k = 0; k < columns; k++) {
            positive_minavg.at(k) += positive_flight_mins.at(k);
            positive_maxavg.at(k) += positive_flight_maxs.at(k);
        }
    }

    int negative_count = 0;
    cout << "calculating negative min max avg" << endl;
    for (int i = 0; i < n_flights2; i++) {
        negative_flight_mins.assign(columns, numeric_limits<double>::max());
        negative_flight_maxs.assign(columns, -numeric_limits<double>::max());

        for (int j = 0; j < rows2[i]; j++) {
            for (int k = 0; k < columns; k++) {
//                cout << "checking data " << i << " " << j << " " << k << endl;

                if (data2[i][j][k] < all_mins.at(k)) all_mins.at(k) = data2[i][j][k];
                if (data2[i][j][k] > all_maxs.at(k)) all_maxs.at(k) = data2[i][j][k];
                negative_avgs.at(k) += data2[i][j][k];
                if (j < rows2[i]-1) negative_roc.at(k) += fabs(data2[i][j][k] - data2[i][j+1][k]);

                if (data2[i][j][k] < negative_flight_mins.at(k)) negative_flight_mins.at(k) = data2[i][j][k];
                if (data2[i][j][k] > negative_flight_maxs.at(k)) negative_flight_maxs.at(k) = data2[i][j][k];
//                cout << "set negative avgs[" << k << "]" << endl;
                negative_count++;
            }
        }

        for (int k = 0; k < columns; k++) {
            negative_minavg.at(k) += negative_flight_mins.at(k);
            negative_maxavg.at(k) += negative_flight_maxs.at(k);
        }
    }

    for (int i = 0; i < columns; i++) {
        positive_avgs[i] /= positive_count;
        positive_roc[i] /= (positive_count - n_flights1);
        positive_minavg.at(i) /= n_flights1;
        positive_maxavg.at(i) /= n_flights1;


        negative_avgs[i] /= negative_count;
        negative_roc[i] /= (negative_count - n_flights2);
        negative_minavg.at(i) /= n_flights2;
        negative_maxavg.at(i) /= n_flights2;
    }

    if (column_headers.size() != columns) {
        cerr << "ERROR: column_headers.size(): " << column_headers.size() << " != columns " << columns << endl;
        exit(1);
    }

    for (int i = 0; i < column_headers.size(); i++) {
        cout << column_headers[i] << endl;
        cout << std::fixed;
        cout << "    with    event overall avg: " << setw(15) << setprecision(10) << positive_avgs[i] << ", avg abs roc: " << setw(15) << setprecision(10) << positive_roc[i] << ", avg min: " << setw(15) << setprecision(10) << positive_minavg[i] << ", avg max: " << positive_maxavg[i] << endl;
        cout << "    without event overall avg: " << setw(15) << setprecision(10) << negative_avgs[i] << ", avg abs roc: " << setw(15) << setprecision(10) << negative_roc[i] << ", avg min: " << setw(15) << setprecision(10) << negative_minavg[i] << ", avg max: " << negative_maxavg[i] << endl;
    }


    for (int i = 0; i < n_flights1; i++) {
        for (int j = 0; j < rows1[i]; j++) {
            for (int k = 0; k < columns; k++) {
                data1[i][j][k] = (data1[i][j][k] - all_mins[k]) / (all_maxs[k] - all_mins[k]);
            }
        }
    }

    for (int i = 0; i < n_flights2; i++) {
        for (int j = 0; j < rows2[i]; j++) {
            for (int k = 0; k < columns; k++) {
                data2[i][j][k] = (data2[i][j][k] - all_mins[k]) / (all_maxs[k] - all_mins[k]);
            }
        }
    }

    int vib_n11_pos, vib_n12_pos;
    for (int i = 0; i < column_headers.size(); i++) {
        if (column_headers[i].compare("VIB_N11") == 0) vib_n11_pos = i;
        if (column_headers[i].compare("VIB_N12") == 0) vib_n12_pos = i;
    }   

    cout << "VIB_N11 at " << vib_n11_pos << ", VIB_N12 at " << vib_n12_pos << endl;

    cout << "calculating covariance and stddev" << endl;

    for (int i = 0; i < columns; i++) {
        positive_avgs[i] = (positive_avgs[i] - all_mins[i]) / (all_maxs[i] - all_mins[i]);
    }

    for (int lag = 1; lag < 6; lag++) {
        int count = 0;
        vector<double> stddev(columns, 0);

        vector<double> covariance_n11(columns, 0);
        vector<double> covariance_n12(columns, 0);

        for (int i = 0; i < n_flights1; i++) {
            for (int j = 0; j < rows1[i] - lag; j++) {
                for (int k = 0; k < columns; k++) {
                    stddev[k] += (data1[i][j][k] - positive_avgs[k]) * (data1[i][j][k] - positive_avgs[k]);
                    count++;

                    if (k == vib_n11_pos) continue;
                    if (k == vib_n12_pos) continue;
                    covariance_n11[k] += (data1[i][j][k] - positive_avgs[k]) * (data1[i][j + lag][vib_n11_pos] - positive_avgs[vib_n11_pos]);
                    covariance_n12[k] += (data1[i][j][k] - positive_avgs[k]) * (data1[i][j + lag][vib_n12_pos] - positive_avgs[vib_n12_pos]);
                }
            }
        }

        for (int i = 0; i < columns; i++) {
            stddev[i] /= count;
        }

        vector<double> correlation_n11(columns, 0);
        vector<double> correlation_n12(columns, 0);
        for (int i = 0; i < columns; i++) {
            if (i == vib_n11_pos) continue;
            if (i == vib_n12_pos) continue;
            correlation_n11[i] = covariance_n11[i] / (stddev[i] * stddev[vib_n11_pos]);
            correlation_n12[i] = covariance_n12[i] / (stddev[i] * stddev[vib_n12_pos]);

            correlation_n11[i] /= count;
            covariance_n11[i] /= count;

            correlation_n12[i] /= count;
            covariance_n12[i] /= count;
        }

        cout << "covariance and correlations for lag: " << lag << endl;
        for (int i = 0; i < column_headers.size(); i++) {
            cout << column_headers[i] << endl;
            cout << std::fixed;
            cout << "    with    event -- covariance n11: " << setw(15) << setprecision(10) << covariance_n11[i] << ", correlation n11: " << setw(15) << setprecision(10) << correlation_n11[i] << ", covariance n12: " << setw(15) << setprecision(10) << covariance_n12[i] << ", correlation n12: " << setw(15) << setprecision(10) << correlation_n12[i] << endl;

            //cout << "    without event overall avg: " << setw(15) << setprecision(10) << negative_avgs[i] << ", avg abs roc: " << setw(15) << setprecision(10) << negative_roc[i] << ", avg min: " << setw(15) << setprecision(10) << negative_minavg[i] << ", avg max: " << negative_maxavg[i] << endl;
        }
    }

}

void normalize_data(double **data, int rows, int columns) {
    //normalize the data
    vector<double> mins(columns, numeric_limits<double>::max());
    vector<double> maxs(columns, -numeric_limits<double>::max());

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            if (data[i][j] < mins[j]) mins[j] = data[i][j];
            if (data[i][j] > maxs[j]) maxs[j] = data[i][j];
        }
    }

    cerr << "#";
    for (int i = 0; i < columns; i++) cerr << " " << mins[i];
    cerr << endl;

    cerr << "#";
    for (int i = 0; i < columns; i++) cerr << " " << maxs[i];
    cerr << endl;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            data[i][j] = (data[i][j] - mins[j]) / (maxs[j] - mins[j]);
        }
    }
}


void get_output_data(double **input_data, int rows, int cols, const vector<string> &input_headers, const vector<string> &output_headers, double ***output_data) {
    /*
    cout << "input_headers.size: " << input_headers.size() << endl;
    cout << "input headers: ";
    for (uint32_t i = 0; i < input_headers.size(); i++) {
        cout << " " << input_headers.at(i);
    }
    cout << endl;

    cout << "output_headers.size: " << output_headers.size() << endl;
    cout << "output headers: ";
    for (uint32_t i = 0; i < output_headers.size(); i++) {
        cout << " " << output_headers.at(i);
    }
    cout << endl;
    */


    vector<uint32_t> output_columns;
    for (uint32_t i = 0; i < output_headers.size(); i++) {
        for (uint32_t j = 0; j < input_headers.size(); j++) {
            if (output_headers.at(i).compare( input_headers.at(j) ) == 0) {
                output_columns.push_back(j);
            }
        }
    }

    if (output_columns.size() != output_headers.size()) {
        cerr << "ERROR: one or more of the output columns not found, output_columns.size(): " << output_columns.size() << endl;
        exit(1);
    }

//    cout << "output data:" << endl;
    (*output_data) = new double*[rows - 1];
    for (uint32_t i = 0; i < rows - 1; i++) {
        (*output_data)[i] = new double[output_columns.size()];

        for (uint32_t j = 0; j < output_columns.size(); j++) {
            (*output_data)[i][j] = input_data[i+1][output_columns.at(j)];
//            cout << " " << (*output_data)[i][j];
        }
//        cout << endl;
    }
}


void write_flight_data(string output_file, const vector<string> &column_names, double **data, int rows, int columns) {
    ofstream *file = new ofstream(output_file.c_str());

    if (!file->is_open()) {
        cerr << "could not open file '" << output_file << "' to store flight data." << endl;
        exit(1);
    }

    (*file) << "TIME";
    for (uint32_t i = 0; i < column_names.size(); i++) {
        (*file) << ", " << column_names[i];
    }
    (*file) << endl;

    for (uint32_t i = 0; i < rows; i++) {
        (*file) << i;
        for (uint32_t j = 0; j < column_names.size(); j++) {
            (*file) << ", " << data[i][j];
        }
        (*file) << endl;
    }

    file->close();
    delete file;
}
