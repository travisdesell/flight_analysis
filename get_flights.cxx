#include <cmath>

#include <fstream>
using std::ofstream;

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <map>
using std::map;

#include <sstream>
using std::ostringstream;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include <stdint.h>

#include <limits>

/**
 *  For MYSQL
 */
#include "mysql.h"

/**
 *  For argument parsing and other helpful functions
 */
#include "tao/undvc_common/arguments.hxx"

#define mysql_query_check(conn, query) __mysql_check (conn, query, __FILE__, __LINE__)

void __mysql_check(MYSQL *conn, string query, const char *file, const int line) {
    mysql_query(conn, query.c_str());

    if (mysql_errno(conn) != 0) {
        ostringstream ex_msg;
        ex_msg << "ERROR in MySQL query: '" << query.c_str() << "'. Error: " << mysql_errno(conn) << " -- '" << mysql_error(conn) << "'. Thrown on " << file << ":" << line;
        cerr << ex_msg.str() << endl;
        exit(1);
    }
}

void write_flight_data(string filename, const vector< vector<string>* > &flight_data, const vector<string> &column_names) {
    if (flight_data.size() == 0) {
        cerr << "Did not write flight, flight_data.size() == 0" << endl;
        return;
    }

    ofstream *file = new ofstream(filename.c_str());

    if (!file->is_open()) {
        cerr << "could not open file '" << filename << "' to store flight data." << endl;
        exit(1);
    }

    (*file) << "#";
    for (uint32_t i = 0; i < column_names.size(); i++) {
        (*file) << " " << column_names[i];
    }
    (*file) << endl;

    vector<double> min(column_names.size(), std::numeric_limits<double>::max());
    vector<double> max(column_names.size(), std::numeric_limits<double>::min());
    vector<double> avg(column_names.size(), 0);
    vector<double> dev(column_names.size(), 0);
    vector<double> min_delta(column_names.size(), std::numeric_limits<double>::max());
    vector<double> max_delta(column_names.size(), std::numeric_limits<double>::min());
    vector<double> avg_delta(column_names.size(), 0);
    vector<double> dev_delta(column_names.size(), 0);

    //get the mins, maxes and sums
    int count = 0;
    for (uint32_t i = 0; i < (flight_data.size() - 1); i++) {
       vector<string> *flight_row = flight_data.at(i);

        //Skip weird times
//       if (atoi(flight_row->at(0).c_str()) % 1000 != 0) continue;

        for (uint32_t j = 0; j < flight_row->size(); j++) {
            double value = atof(flight_row->at(j).c_str());
            if (value < min[j]) min[j] = value;
            if (value > max[j]) max[j] = value;
            avg[j] += value;

            double delta = atof( flight_data.at(i+1)->at(j).c_str() ) - value;
            if (delta < min_delta[j]) min_delta[j] = delta;
            if (delta > max_delta[j]) max_delta[j] = delta;
            avg_delta[j] += delta;
        }

        count++;
    }

    //calculate the averages
    for (uint32_t j = 0; j < column_names.size(); j++) {
        avg[j] /= count;
        avg_delta[j] /= count;
    }

    //sum the devations
    for (uint32_t i = 0; i < (flight_data.size() - 1); i++) {
       vector<string> *flight_row = flight_data.at(i);

        //Skip weird times
//       if (atoi(flight_row->at(0).c_str()) % 1000 != 0) continue;

        for (uint32_t j = 0; j < flight_row->size(); j++) {
            double value = atof(flight_row->at(j).c_str());
            dev[j] += sqrt((avg[j] - value) * (avg[j] - value));

            double delta = value - atof( flight_data.at(i+1)->at(j).c_str() );
            dev_delta[j] += sqrt((avg_delta[j] - delta) * (avg_delta[j] - delta));
        }
    }

    //calculate the deviations
    for (uint32_t j = 0; j < column_names.size(); j++) {
        dev[j] /= count;
        dev_delta[j] /= count;
    }

    (*file) << "#min";
    for (uint32_t j = 0; j < column_names.size(); j++) (*file) << " " << min[j];
    (*file) << endl;

    (*file) << "#max";
    for (uint32_t j = 0; j < column_names.size(); j++) (*file) << " " << max[j];
    (*file) << endl;

    (*file) << "#avg";
    for (uint32_t j = 0; j < column_names.size(); j++) (*file) << " " << avg[j];
    (*file) << endl;

    (*file) << "#dev";
    for (uint32_t j = 0; j < column_names.size(); j++) (*file) << " " << dev[j];
    (*file) << endl;

    (*file) << "#min delta";
    for (uint32_t j = 0; j < column_names.size(); j++) (*file) << " " << min_delta[j];
    (*file) << endl;

    (*file) << "#max delta";
    for (uint32_t j = 0; j < column_names.size(); j++) (*file) << " " << max_delta[j];
    (*file) << endl;

    (*file) << "#avg delta";
    for (uint32_t j = 0; j < column_names.size(); j++) (*file) << " " << avg_delta[j];
    (*file) << endl;

    (*file) << "#dev delta";
    for (uint32_t j = 0; j < column_names.size(); j++) (*file) << " " << dev_delta[j];
    (*file) << endl;


 
    for (uint32_t i = 0; i < flight_data.size(); i++) {
       vector<string> *flight_row = flight_data.at(i);

        //Skip weird times
//       if (atoi(flight_row->at(0).c_str()) % 1000 != 0) continue;

        for (uint32_t j = 0; j < flight_row->size(); j++) {
            //normalize the flight data between 0 and 1
            (*file) << " " << ((atof(flight_row->at(j).c_str()) - min[j]) / (max[j] - min[j]));
        }
        (*file) << endl;
    }

    delete file;
}

bool column_exists(map<string,int> column_name_map, string name) {
    return column_name_map.find(name) != column_name_map.end();
}

template <typename T>
string to_string(T value) {
    ostringstream oss;
    oss << value;
    return oss.str();
}

int main(int argc /* number of command line arguments */, char **argv /* command line argumens */) {
    vector<string> arguments(argv, argv + argc);

    MYSQL *conn = mysql_init(NULL);

    if (conn == NULL) {
        cerr << "Error initializing mysql: " << mysql_errno(conn) << ", '" << mysql_error(conn) << "'" << endl;
        exit(1);
    }

    string db_host, db_name, db_password, db_user;
    get_argument(arguments, "--db_host", true, db_host);
    get_argument(arguments, "--db_name", true, db_name);
    get_argument(arguments, "--db_user", true, db_user);
    get_argument(arguments, "--db_password", true, db_password);

    //shoud get database info from a file

    if (mysql_real_connect(conn, db_host.c_str(), db_user.c_str(), db_password.c_str(), db_name.c_str(), 0, NULL, 0) == NULL) {
        cerr << "Error connecting to database: " << mysql_errno(conn) << ", '" << mysql_error(conn) << "'" << endl;
        exit(1);
    }

    bool random_selection = argument_exists(arguments, "--random");

    int fleet_id = 0;
    get_argument(arguments, "--fleet_id", true, fleet_id);

    //UND's fleet is fleet id 1. This will select 5000 random flight ids from the database.
    ostringstream flight_id_query;

    if (argument_exists(arguments, "--excessive_roll")) {
        flight_id_query << "SELECT DISTINCT(main.flight) FROM main, flight_id WHERE roll_attitude NOT BETWEEN -60 AND 60 AND main.flight=flight_id.id AND flight_id.fleet_id=1";

    } else if (argument_exists(arguments, "--excessive_pitch")) {
        flight_id_query << "SELECT DISTINCT(main.flight) FROM main, flight_id WHERE pitch_attitude NOT BETWEEN -30 AND 30 AND main.flight=flight_id.id AND flight_id.fleet_id=1";

    } else if (argument_exists(arguments, "--excessive_speed")) {
        flight_id_query << "SELECT DISTINCT(main.flight) FROM main, flight_id WHERE indicated_airspeed > 163 AND main.flight=flight_id.id AND flight_id.fleet_id=1";

    } else if (argument_exists(arguments, "--high_cht")) {
        flight_id_query << "SELECT DISTINCT(main.flight) FROM main, flight_id WHERE (eng_1_cht_1 > 500 OR eng_1_cht_2 > 500 OR eng_1_cht_3 > 500 OR eng_1_cht_4 > 500) AND main.flight=flight_id.id AND flight_id.fleet_id=1";

    } else if (argument_exists(arguments, "--high_altitude")) {
        flight_id_query << "SELECT DISTINCT(main.flight) FROM main, flight_id WHERE msl_altitude > 12800 AND main.flight=flight_id.id AND flight_id.fleet_id=1";

    } else if (argument_exists(arguments, "--low_fuel")) {
        cerr << "WARNING: query is potentially very slow." << endl;
        flight_id_query << "SELECT distinct(flight) FROM main, flight_id WHERE (fuel_quantity_left_main + fuel_quantity_right_main) < 8 AND indicated_airspeed > 30 AND main.flight=flight_id.id AND flight_id.fleet_id=1";

    } else if (argument_exists(arguments, "--low_oil_pressure")) {
        cerr << "WARNING: query is potentially very slow." << endl;
        flight_id_query << "SELECT DISTINCT(main.flight) FROM main, flight_id WHERE eng_1_oil_press < 20 AND indicated_airspeed > 30 AND main.flight=flight_id.id AND flight_id.fleet_id=1";

    } else {
        flight_id_query << "SELECT id FROM flight_id WHERE fleet_id = " << fleet_id;
    }

    int max_number_flights = 0;
    get_argument(arguments, "--max_number_flights", true, max_number_flights);

    int min_flight_id = -1, max_flight_id = -1;
    get_argument(arguments, "--min_flight_id", false, min_flight_id);
    get_argument(arguments, "--max_flight_id", false, max_flight_id);

    if (min_flight_id > 0 && max_flight_id > 0) {
        flight_id_query << " AND id BETWEEN " << min_flight_id << " AND " << max_flight_id;
    }

    if (random_selection) flight_id_query << " ORDER BY rand()";

    flight_id_query << " LIMIT " << max_number_flights;

    cout << "query: '" << flight_id_query.str() << "'" << endl;


    mysql_query_check(conn, flight_id_query.str());
    MYSQL_RES *flight_id_result = mysql_store_result(conn);


    long excessive_roll_flights = 0;
    long excessive_pitch_flights = 0;
    long excessive_speed_flights = 0;
    long high_cht_flights = 0;
    long high_altitude_flights = 0;
    long low_fuel_flights = 0;
    long low_oil_pressure_flights = 0;

    string output_directory = "./";
    get_argument(arguments, "--output_directory", false, output_directory);

    cout << "using '" << output_directory << "' as the output directory, this can be changed with the --output_directory command line argument" << endl;

    MYSQL_ROW flight_id_row;
    while ((flight_id_row = mysql_fetch_row(flight_id_result)) != NULL) {
        int flight_id = atoi(flight_id_row[0]);

        cout << "Processing flight " << flight_id << endl;

        ostringstream flight_data_query;
//        flight_data_query << "SELECT * FROM main WHERE flight ='" << flight_id << "'";
//        flight_data_query << "SELECT id, flight, phase, time, msl_altitude, indicated_airspeed, vertical_airspeed, tas, heading, course, pitch_attitude, roll_attitude, eng_1_rpm, vertical_acceleration, longitudinal_acceleration, lateral_acceleration, oat, groundspeed, latitude, longitude, nav_1_freq, nav_2_freq, obs_1, fuel_quantity_left_main, fuel_quantity_right_main, eng_1_fuel_flow, eng_1_oil_press, eng_1_oil_temp, eng_1_cht_1, eng_1_cht_2, eng_1_cht_3, eng_1_cht_4, eng_1_egt_1, eng_1_egt_2, eng_1_egt_3, eng_1_egt_4, system_1_volts, system_2_volts, system_1_amps, system_2_amps FROM main WHERE flight ='" << flight_id << "'";

//        flight_data_query << "SELECT roll_attitude, pitch_attitude, indicated_airspeed, eng_1_cht_1, eng_1_cht_2, eng_1_cht_3, eng_1_cht_4, msl_altitude, fuel_quantity_left_main, fuel_quantity_right_main, eng_1_oil_press FROM main WHERE flight = '" << flight_id << "'" << endl;

        flight_data_query << "SELECT time, roll_attitude, pitch_attitude, indicated_airspeed, msl_altitude FROM main WHERE flight = '" << flight_id << "'" << endl;

        mysql_query_check(conn, flight_data_query.str());
        MYSQL_RES *flight_data_result = mysql_store_result(conn);

        long current_flight_data_row = 0;

        long excessive_roll_excedence_count = 0;
        long excessive_pitch_excedence_count = 0;
        long excessive_speed_excedence_count = 0;
        long high_cht_excedence_count = 0;
        long high_altitude_excedence_count = 0;
        long low_fuel_excedence_count = 0;
        long low_oil_pressure_excedence_count = 0;

        map<string,int> column_name_map;
        vector<string> column_names;

        bool had_null = false;

        int field_count = 0;
        vector< vector<string>* > flight_data;
        //Need to clean this up after processing a flight

        MYSQL_ROW flight_data_row;
        while ((flight_data_row = mysql_fetch_row(flight_data_result)) != NULL) {

            if (current_flight_data_row == 0) {
                field_count = mysql_num_fields(flight_data_result);

                MYSQL_FIELD *fields = mysql_fetch_fields(flight_data_result);
                for (int i = 0; i < field_count; i++) {
                    if (flight_data_row[i] == NULL) {
                        cerr << "Error, flight data row[" << i << "] is null." << endl;
                        exit(1);
//                        continue;
                    }

                    column_name_map[ fields[i].name ] = i;
                    column_names.push_back( fields[i].name );
                    cout << "field[" << i << "]: " << fields[i].name << endl;
                }
            }

            vector<string> *flight_row = new vector<string>(field_count, "");

            had_null = false;
            for (int i = 0; i < field_count; i++) {
                if (flight_data_row[i] == NULL) {

                    cerr << endl;
                    cerr << " Error, flight data row[" << column_names[i] << "] (row number " << i << ") is null." << endl;
//                    continue;

                    flight_row->at(i) = "NULL";
                    had_null = true;
                } else {
                    flight_row->at(i) = flight_data_row[i];
                }
            }
            flight_data.push_back(flight_row);

            current_flight_data_row++;

            if (had_null) continue;


            if (column_exists(column_name_map, "roll_attitude") && ((atof(flight_data_row[ column_name_map.at("roll_attitude") ]) > 60) || (atof(flight_data_row[ column_name_map.at("roll_attitude") ]) < -60))) {
                //excessive roll
                excessive_roll_excedence_count++;
            }

            if (column_exists(column_name_map, "pitch_attitude") && ((atof(flight_data_row[ column_name_map.at("pitch_attitude") ]) > 30) || (atof(flight_data_row[ column_name_map.at("pitch_attitude") ]) < -30))) {
                //excessive pitch
                excessive_pitch_excedence_count++;
            }

            if (column_exists(column_name_map, "indicated_airspeed") && atof(flight_data_row[ column_name_map.at("indicated_airspeed") ]) > 163) {
                //excessive speed
                excessive_speed_excedence_count++;
            }

            if (    (column_exists(column_name_map, "eng_1_cht_1") && atof(flight_data_row[ column_name_map.at("eng_1_cht_1") ]) > 500) ||
                    (column_exists(column_name_map, "eng_1_cht_2") && atof(flight_data_row[ column_name_map.at("eng_1_cht_2") ]) > 500) ||
                    (column_exists(column_name_map, "eng_1_cht_3") && atof(flight_data_row[ column_name_map.at("eng_1_cht_3") ]) > 500) ||
                    (column_exists(column_name_map, "eng_1_cht_4") && atof(flight_data_row[ column_name_map.at("eng_1_cht_4") ]) > 500)) {
                //high CHT 1
                high_cht_excedence_count++;
            }

            if (column_exists(column_name_map, "msl_altitude") && atof(flight_data_row[ column_name_map.at("msl_altitude") ]) > 12800) {
                //high altitude
                high_altitude_excedence_count++;
            }

            if (column_exists(column_name_map, "fuel_quantity_right_main") && column_exists(column_name_map, "fuel_quantity_left_main") && column_exists(column_name_map, "indicated_airspeed") && 
                    ((atof(flight_data_row[ column_name_map.at("fuel_quantity_left_main") ]) + atof(flight_data_row[ column_name_map.at("fuel_quantity_right_main") ]) < 8) &&
                     (atof(flight_data_row[ column_name_map.at("indicated_airspeed") ]) > 30))) {
                //Low Fuel
                low_fuel_excedence_count++;
            }

            if ((column_exists(column_name_map, "eng_1_oil_press") && atof(flight_data_row[ column_name_map.at("eng_1_oil_press") ]) < 20) &&
                    (column_exists(column_name_map, "indicated_airspeed") && atof(flight_data_row[ column_name_map.at("indicated_airspeed") ]) > 30)) {
                //Low oil pressure
                low_oil_pressure_excedence_count++;
            }
        }

        if (flight_data.size() > 0) {
            //The last rows in the database seem to be messed up, so remove the last row.
            vector<string> *flight_row = flight_data.back();
            flight_data.pop_back();
            delete flight_row;
        }

        cout << "Number of excessive roll excedences:     " << excessive_roll_excedence_count << endl;
        cout << "Number of excessive pitch excedences:    " << excessive_pitch_excedence_count << endl;
        cout << "Number of excessive speed excedences:    " << excessive_speed_excedence_count << endl;
        cout << "Number of high CHT 1 excedences:         " << high_cht_excedence_count << endl;
        cout << "Number of high altitude excedences:      " << high_altitude_excedence_count << endl;
        cout << "Number of low fuel excedences:           " << low_fuel_excedence_count << endl;
        cout << "Number of low oil pressure excedences:   " << low_oil_pressure_excedence_count << endl;
        cout << endl;


        if (excessive_roll_excedence_count > 0) {
            excessive_roll_flights++;
            write_flight_data(output_directory + "excessive_roll/" + to_string(flight_id), flight_data, column_names);
        }

        if (excessive_pitch_excedence_count > 0) {
            excessive_pitch_flights++;
            write_flight_data(output_directory + "excessive_pitch/" + to_string(flight_id), flight_data, column_names);
        }

        if (excessive_speed_excedence_count > 0) {
            excessive_speed_flights++;
            write_flight_data(output_directory + "excessive_speed/" + to_string(flight_id), flight_data, column_names);
        }

        if (high_cht_excedence_count > 0) {
            high_cht_flights++;
            write_flight_data(output_directory + "high_cht/" + to_string(flight_id), flight_data, column_names);
        }

        if (high_altitude_excedence_count > 0) {
            high_altitude_flights++;
            write_flight_data(output_directory + "high_altitude/" + to_string(flight_id), flight_data, column_names);
        }

        if (low_fuel_excedence_count > 0) {
            low_fuel_flights++;
            write_flight_data(output_directory + "low_fuel/" + to_string(flight_id), flight_data, column_names);
        }

        if (low_oil_pressure_excedence_count > 0) {
            low_oil_pressure_flights++;
            write_flight_data(output_directory + "low_oil_pressure/" + to_string(flight_id), flight_data, column_names);
        }

        if (excessive_roll_excedence_count == 0 && excessive_pitch_excedence_count == 0 && excessive_speed_excedence_count == 0 &&
            high_cht_excedence_count == 0 && high_altitude_excedence_count == 0 && low_fuel_excedence_count == 0 && low_oil_pressure_excedence_count == 0) {
            write_flight_data(output_directory + "no_excedence/" + to_string(flight_id), flight_data, column_names);
        }

        while ( !flight_data.empty() ) {
            vector<string> *flight_row = flight_data.back();

            /*
            for (int i = 0; i < field_count; i++) {
                cout << " " << flight_row[i];
            }
            cout << endl;
            */

            flight_data.pop_back();
            delete flight_row;
        }

        mysql_free_result(flight_data_result);
    }

    cout << "Number of excessive roll flights:     " << excessive_roll_flights << endl;
    cout << "Number of excessive pitch flights:    " << excessive_pitch_flights << endl;
    cout << "Number of excessive speed flights:    " << excessive_speed_flights << endl;
    cout << "Number of high CHT 1 flights:         " << high_cht_flights << endl;
    cout << "Number of high altitude flights:      " << high_altitude_flights << endl;
    cout << "Number of low fuel flights:           " << low_fuel_flights << endl;
    cout << "Number of low oil pressure flights:   " << low_oil_pressure_flights << endl;
    cout << endl;

    mysql_free_result(flight_id_result);
}
