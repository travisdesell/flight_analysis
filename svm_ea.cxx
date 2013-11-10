#include <fstream>
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

#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
using boost::variate_generator;
using boost::mt19937;
//using boost::exponential_distribution;
//using boost::gamma_distribution;
using boost::uniform_real;


#include <boost/tokenizer.hpp>
using boost::tokenizer;
using boost::char_separator;

/**
 *  For argument parsing and other helpful functions
 */
#include "undvc_common/arguments.hxx"


/****
	*	Assuming tol = threshold
	*	eps = e
 ****/

typedef double (*KernelFunctionType)(const vector<double> &, const vector<double> &);

KernelFunctionType kernel_function = NULL;

double threshold = 0;
double C = 5;
double e = 0.0001;

double kernel_gamma = 0.5;
double kernel_r = 0;
double kernel_d = 2;

bool verbose = false;
bool complete = false;
bool modified = true;

long steps = 0;

variate_generator<mt19937, uniform_real<> > random_0_1( mt19937(time(0)), uniform_real<>(0.0, 1.0));

//Transpose multiply  = p1'p2; or <p1,p2>
double transpose_multiply(const vector<double> &p1, const vector<double> &p2) {
    double result = 0;
    for (uint32_t i = 0; i < p1.size(); i++) result += p1[i] * p2[i];
    return result;
}

double linear_kernel(const vector<double> &point1, const vector<double> &point2) {
    return transpose_multiply(point1, point2);
}

double polynomial_kernel(const vector<double> &point1, const vector<double> &point2) {
    return pow(kernel_gamma + transpose_multiply(point1, point2) + kernel_r, kernel_d);
}

double rbf_kernel(const vector<double> &point1, const vector<double> &point2) {
    return exp(-kernel_gamma * transpose_multiply(point1, point1) - (2 * transpose_multiply(point1, point2)) + transpose_multiply(point2, point2));
}

class TrainingExample {
    public:
        const int id;
        const string type;
        const int desired_output;

        const vector<double> training_point;

        double estimated_error;
        double lagrange_multiplier;

        TrainingExample(int _id, string _type, int _desired_output, const vector<double> &_training_point) : id(_id), type(_type), desired_output(_desired_output), training_point(_training_point) {
            estimated_error = 0.0;
            lagrange_multiplier = 0.0;
        }

        bool at_bound(double C) {
            return (lagrange_multiplier == C || lagrange_multiplier == 0);
        }

        bool equals(TrainingExample *other) {
            return id == other->id;
        }
};

vector< TrainingExample* > training_examples;

void read_examples_from_file(string input_filename) {
    cout << "loading training data from: " << input_filename << endl;

    /****
     *	Datafile in format input1,input2,...,inputn,desired_output
     ****/

    long positive_examples = 0;
    long negative_examples = 0;

    ifstream input_file(input_filename.c_str());

    string line;
    int count = 0;

    while (input_file.good()) {
        getline(input_file, line);

        if (!input_file.good()) break;

        vector<string> data_row;

        char_separator<char> sep(" ", "");
        tokenizer<char_separator<char> > tok(line, sep);
        tokenizer<char_separator<char> >::iterator i = tok.begin();

        int id = lexical_cast<int>( *i );
        ++i;
//        cout << "parsed id: " << id << endl;

        string type = *i;
        ++i;
//        cout << "parsed type: " << type << endl;

        int desired_output = lexical_cast<int>( *i );
        ++i;
        if (desired_output == 0) desired_output = -1;
//        cout << "parsed desired output: " << desired_output << endl;

        vector<double> training_point;
        for (; i != tok.end(); ++i) {
            training_point.push_back( lexical_cast<double>(*i) );
//            cout << " " << *i;
        }
//        cout << endl;

        TrainingExample *training_example = new TrainingExample(id, type, desired_output, training_point);

        bool found = false;
        for (uint32_t i = 0; i < training_examples.size(); i++) {    //this could be done more efficiently
            if (training_examples[i]->id == id) {
                found = true;
                break;
            }
        }

        if (!found) {
            training_examples.push_back(training_example);

            if (desired_output == 1) {
                positive_examples++;
            } else {
                negative_examples++;
            }

            count++;
        } else {
            delete training_example;
        }
    }
    input_file.close();

    cout << "file: " << input_filename << " sucessfully loaded." << endl;
    cout << "\t" << positive_examples << " positive examples." << endl;
    cout << "\t" << negative_examples << " negative examples." << endl;
}

double get_output(const vector<double> &input) {
    double result = 0;
    for (uint32_t i = 0; i < training_examples.size() ; i++) {
        TrainingExample* current = training_examples[i];
        result += current->lagrange_multiplier * current->desired_output * kernel_function(current->training_point, input);
    }
    return result - threshold;
}

void test_svm() {
    long misclassified_negative = 0;
    long total_negative = 0;

    long misclassified_positive = 0;
    long total_positive = 0;


    cout << "lagrange multipliers: " << endl;
    for (uint32_t i = 0; i < training_examples.size(); i++) {
        cout << "\t" << training_examples[i]->lagrange_multiplier << endl;
    }


    for (uint32_t i = 0; i < training_examples.size(); i++) {
        if (training_examples[i]->desired_output < 0) total_negative++;
        if (training_examples[i]->desired_output > 0) total_positive++;

        if (training_examples[i]->desired_output < 0 && get_output(training_examples[i]->training_point) > 0) {
//            cout << "misclassified negative: " << training_examples[i]->id << " -- desired output " << training_examples[i]->desired_output << ", scored " << get_output(training_examples[i]->training_point) << endl;
            misclassified_negative++;
        }

        if (training_examples[i]->desired_output > 0 && get_output(training_examples[i]->training_point) < 0) {
//            cout << "misclassified positive: " << training_examples[i]->id << " -- desired output " << training_examples[i]->desired_output << ", scored " << get_output(training_examples[i]->training_point) << endl;
            misclassified_positive++;
        }
    }

    cerr << (misclassified_negative + misclassified_positive) << " of " << training_examples.size() << " examples misclassified." << endl;
    cerr << misclassified_negative << " of " << total_negative << " classified as positive when desired output was negative." << endl;
    cerr << misclassified_positive << " of " << total_positive << " classified as negative when desired output was positive." << endl;
}


double fitness(const vector<double> &lagrange_multipliers) {
    double to_maximize = 0;
    for (uint32_t i = 0; i < training_examples.size(); i++) {
        to_maximize += training_examples[i]->lagrange_multiplier;

        for (uint32_t j = 0; j < training_examples.size(); j++) {
            to_maximize -= 0.5 * training_examples[i]->lagrange_multiplier * training_examples[j]->lagrange_multiplier * training_examples[i]->desired_output * training_examples[j]->desired_output * kernel_function(training_examples[i]->training_point, training_examples[j]->training_point);
        }
    }

    cout << "to maximize: " << to_maximize << endl;
}

int main(int argc, char** argv) {
    vector<string> arguments(argv, argv + argc);

    get_argument(arguments, "--threshold", false, threshold);
    get_argument(arguments, "--C", false, C);
    get_argument(arguments, "--e", false, e);
    verbose = argument_exists(arguments, "--verbose");

    cout << "threshold = " << threshold << endl;
    cout << "e = " << e << endl;
    cout << "C = " << C << endl;
    cout << "verbose = " << verbose << endl;

    string training_file = "";
    string testing_file = "";

    get_argument(arguments, "--training_file", true, training_file);
    get_argument(arguments, "--training_file", false, testing_file);

    string kernel_type;
    get_argument(arguments, "--kernel_type", true, kernel_type);

    if (kernel_type.compare("linear")) {
        kernel_function = linear_kernel;
    } else if (kernel_type.compare("poly")) {
        if (!get_argument(arguments, "--gamma", false, kernel_gamma)) cerr << "'--gamma <F>' not found, using default value: " << kernel_gamma << endl;
        if (!get_argument(arguments, "--r", false, kernel_r)) cerr << "'--r <F>' not found, using default value: " << kernel_r << endl;
        if (!get_argument(arguments, "--d", false, kernel_d)) cerr << "'--d <F>' not found, using default value: " << kernel_d << endl;


        kernel_function = polynomial_kernel;
    } else if (kernel_type.compare("rbf")) {
        if (!get_argument(arguments, "--gamma", false, kernel_gamma)) cerr << "'--gamma <F>' not found, using default value: " << kernel_gamma << endl;

        kernel_function = rbf_kernel;
    } else {
        cerr << "Unknown kernel type specified: '" << kernel_type << "'." << endl;
        cerr << "Possibilities are: lineaar, poly, rbf" << endl;
        exit(1);
    }
    

    read_examples_from_file(training_file);
    train();
    test_svm();
}


/*
	int test_size = 3000;
	int[] non_bound_values = new int[test_size];
	int min = 0;
	int max = 0;

	public void test_svm(String input_file) {
		System.out.println("testing svm with file: " + input_file);

		int count = 0;
		int correct = 0;
		try {
			BufferedReader in = new BufferedReader(new FileReader(input_file));
			String line;

			while ((line = in.readLine()) != null) {
				StringTokenizer t = new StringTokenizer(line, ",");
				int token_count = t.countTokens();
				double[] input = new double[token_count-1];
				if (verbose) System.out.print("testing: ");
				for (int i = 0; i < token_count-1; i++) {
					input[i] = Double.valueOf(t.nextToken()).doubleValue();
					if (verbose) {
						System.out.print(input[i]);
						if (i+1 != token_count-1) System.out.print(", ");
					}
				}
				double output = get_output(input);

				double desired_output = Double.valueOf(t.nextToken()).doubleValue();
				if (desired_output == 0) desired_output = -1;

				if (verbose && output < 0) System.out.print(", output: -1");
				else if (verbose && output > 0) System.out.print(", output: +1");
				else if (verbose) System.out.print(", output: 0");
				
				if (verbose) System.out.println(", desired_output: " + desired_output);

				if (desired_output == -1 && output < 0) correct++;
				else if (desired_output == 1 && output > 0) correct++;
				count++;

				if (count%500 == 0) System.err.println(count + " examples tested.");
			}
			in.close();
		} catch(Exception e) {
			System.err.println("Couldn't read file: " + input_file);
			System.err.println("\tException: " + e);
		}
		System.out.println("SVM testing completed successfully.");
		System.out.println("\t" + correct + " out of " + count + " (" + (100*((double)correct/(double)count)) + "%) estimated correctly.");
		System.out.println("\tTrained in: " + ((double)(end_training_time-begin_training_time)/(double)1000) + "s.");
	}
}
*/
