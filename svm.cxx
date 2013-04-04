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

int non_bound() {
    int non_bound = 0;
    for (uint32_t i = 0; i < training_examples.size(); i++) {
        if ( !training_examples[i]->at_bound(C) ) non_bound++;
    }
    return non_bound;
}

void add_non_bound(int non_bound) {
//		non_bound_values[(int)(steps%test_size)] = non_bound;
//		min = non_bound_values[0];
//		max = non_bound_values[0];
//		for (int i = 1; i < test_size; i++) {
//			if (min > non_bound_values[i]) min = non_bound_values[i];
//			if (max < non_bound_values[i]) max = non_bound_values[i];			
//		}
//		if (steps > test_size && (max-min) < 3) complete = true;
}


void test_svm() {
    long misclassified_negative = 0;
    long total_negative = 0;

    long misclassified_positive = 0;
    long total_positive = 0;

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



bool take_step(TrainingExample* example1, TrainingExample* example2) {
    if (example1->equals(example2)) {
        modified = false;
        return false;
    }
    if (modified) example1->estimated_error = get_output(example1->training_point) - example1->desired_output;

    /****
     *	Calculate L and H
     ****/
    double L, H;

    if (example1->desired_output != example2->desired_output) {
        double difference = example2->lagrange_multiplier - example1->lagrange_multiplier;

        if (0 > difference) L = 0;
        else L = difference;

        if (C < difference + C) H = C;
        else H = difference + C;
    } else {
        double sum = example2->lagrange_multiplier + example1->lagrange_multiplier;

        if (0 > sum - C) L = 0;
        else L = sum - C;

        if (C < sum) H = C;
        else H = sum;
    }

    if (L == H) {
        modified = false;
        return false;
    }

    /****
     *	calculate the new lagrange multipliers
     ****/
    double k11 = kernel_function(example1->training_point, example1->training_point);
    double k12 = kernel_function(example1->training_point, example2->training_point);
    double k22 = kernel_function(example2->training_point, example2->training_point);
    double eta = (2 * k12) - k11 - k22;

    double a1, a2;
    if (eta < 0) {
        a2 = example2->lagrange_multiplier - example2->desired_output * (example1->estimated_error - example2->estimated_error) / eta;

        if (a2 < L) a2 = L;
        else if (a2 > H) a2 = H;
    } else {
        double s = example1->desired_output * example2->desired_output;
        double iy = example1->lagrange_multiplier + s * example2->lagrange_multiplier;

        /****
         *	Find v1 and v2
         ****/

        double v1 = get_output(example1->training_point) + threshold - (example1->desired_output * example1->lagrange_multiplier * k11) - (example2->desired_output * example2->lagrange_multiplier * k12);
        double v2 = get_output(example2->training_point) + threshold - (example1->desired_output * example1->lagrange_multiplier * k12) - (example2->desired_output * example2->lagrange_multiplier * k22);


        /****
         *	find objective values of L and H
         ****/

        double L_temp = iy - s*L;
        double H_temp = iy - s*H;

        double L_objective =	L_temp + L - (0.5 * k11 * L_temp * L_temp) - (0.5 * k22 * L * L) - (s * k12 * L_temp * L) - (example1->desired_output * L_temp * v1) - (example2->desired_output * L * v2);
        double H_objective =	H_temp + H - (0.5 * k11 * H_temp * H_temp) - (0.5 * k22 * H * H) - (s * k12 * H_temp * H) - (example1->desired_output * H_temp * v1) - (example2->desired_output * H * v2);

        if (L_objective > H_objective + e) a2 = L;
        else if (L_objective < H_objective - e) a2 = H;
        else a2 = example2->lagrange_multiplier;
    }

    /****
     *	round to 0 or C if very close
     ****/

    if (a2 < 1e-8) a2 = 0;
    else if (a2 > (C - 1e-8) ) a2 = C;

    if (fabs(a2 - example2->lagrange_multiplier) < e * (a2 + example2->lagrange_multiplier + e)) {
        modified = false;
        return false;
    }

    steps++;
    int nb = non_bound();
    if (steps % 500 == 0) {
        cerr << steps << " steps taken [" << nb << " of " << training_examples.size() << " not at bound]." << endl;
        test_svm();
    }
    add_non_bound(nb);

    a1 = example1->lagrange_multiplier + (example1->desired_output * example2->desired_output) * (example2->lagrange_multiplier - a2);

    /********
     *	Update threshold to reflect change in Lagrange multipliers
     ********/

    double temp1 = example1->desired_output * (a1 - example1->lagrange_multiplier);
    double temp2 = example2->desired_output * (a2 - example2->lagrange_multiplier);
    double b1 = example1->estimated_error + (temp1 * k11) + (temp2 * k12) + threshold;
    double b2 = example2->estimated_error + (temp1 * k12) + (temp2 * k22) + threshold;
    double new_threshold;

    if (!example1->at_bound(C)) new_threshold = b1;
    else if (!example2->at_bound(C)) new_threshold = b2;
    else if (L != H) new_threshold = (b1 + b2) / 2;
    else new_threshold = threshold;

    //Update weight vector to reflect change in a1 & a2, if linear SVM
    //************************************************************************************************************
    //not needed yet
    //************************************************************************************************************


    /********
     *	Update error cache using new Lagrange multipliers
     ********/

    for (uint32_t i = 0; i < training_examples.size(); i++) {
        TrainingExample* current = training_examples[i];

        if (!current->equals(example1) && !current->equals(example2) && !current->at_bound(C)) {
            current->estimated_error +=	(temp1 * kernel_function(example1->training_point, current->training_point)) + (temp2 * kernel_function(example2->training_point, current->training_point)) + threshold - new_threshold;
        }
    }
    /********
     *	Set estimated error to 0
     ********/

    example1->estimated_error = 0;
    example2->estimated_error = 0;

    /********
     *	Store changes
     ********/

    example1->lagrange_multiplier = a1;
    example2->lagrange_multiplier = a2;
    threshold = new_threshold;

    modified = true;
    return true;
}


TrainingExample* get_2nd_example(TrainingExample *first) {
    TrainingExample* target = training_examples[0];
    TrainingExample* current = NULL;

    if (first->estimated_error > 0) {
        /****
         *	if first.estimated_error is positive, choose minimum error example
         ****/
        for (uint32_t i = 0; i < training_examples.size(); i++) {
            current = training_examples[i];
            if (!first->equals(current)) {
                if (target->estimated_error > current->estimated_error) target = current;
            }
        }
    } else {
        /****
         *	if first.estimated_error is negative, choose maximum error example
         ****/
        for (uint32_t i = 0; i < training_examples.size(); i++) {
            current = training_examples[i];
            if (!first->equals(current)) {
                if (target->estimated_error < current->estimated_error) target = current;
            }
        }
    }

    if (current == NULL) {
        cerr << "Error in 2nd choice heuristic, no valid choice" << endl;
        exit(1);
    }
    return current;
}


int examine_example(TrainingExample* example2) {
    example2->estimated_error = get_output(example2->training_point) - example2->desired_output;

    double r2 = (example2->estimated_error - example2->desired_output) * example2->desired_output;

    if ((r2 < -threshold && example2->lagrange_multiplier < C) || (r2 > threshold && example2->lagrange_multiplier > 0)) {
        /********
         *	Find number of non-zero and non-C alpha
         ********/

        if (non_bound() > 1) {
            TrainingExample* example1 = get_2nd_example(example2);

            if (take_step(example1, example2)) return 1;
        }

        int first_example = (int)(random_0_1()*(double)training_examples.size());
        for (uint32_t i = 0; i < training_examples.size(); i++) {
            TrainingExample* example1 = training_examples[first_example % training_examples.size()];

            if (!example1->at_bound(C)) {
                if (take_step(example1, example2)) return 1;
            }

            first_example++;
        }

        first_example = (int)(random_0_1()*(double)training_examples.size());
        for (uint32_t i = 0; i < training_examples.size(); i++) {
            TrainingExample* example1 = training_examples[first_example % training_examples.size()];

            if (take_step(example1, example2)) return 1;
            first_example++;
        }
    }
    return 0;
}


void train() {
    int number_changed = 0;
    bool examine_all = true;

    cout << "Training SVM." << endl;

    long begin_training_time = time(NULL);

    while ((number_changed > 0 || examine_all) && !complete) {
        number_changed = 0;

        if (examine_all) {
            if (verbose) {
                cout << "Examining all examples." << endl;
//                cout << "changed > 0:" << endl;
            }

//            cout << "\t";
            for (uint32_t i = 0; i < training_examples.size(); i++) {
                int changed = examine_example(training_examples[i]);
//                if (changed > 0) cout << setw(8) << training_examples[i]->id;
                number_changed += changed;
            }
//            cout << endl;

        } else {
            if (verbose) {
                cout << "Examining all non-bound examples." << endl;
//                cout << "changed > 0:" << endl;
            }

//            cout << "\t";
            for (uint32_t j = 0; j < training_examples.size(); j++) {
                if (!training_examples[j]->at_bound(C)) {
                    int changed = examine_example(training_examples[j]);
//                    if (changed > 0) cout << setw(8) << training_examples[j]->id;
                    number_changed += changed;
                }
            }
//            cout << endl;
        }

        cout << "number changed: " << number_changed << endl;

        /*
        double to_maximize = 0;
        for (uint32_t i = 0; i < training_examples.size(); i++) {
            to_maximize += training_examples[i]->lagrange_multiplier;

            for (uint32_t j = 0; j < training_examples.size(); j++) {
                to_maximize -= 0.5 * training_examples[i]->lagrange_multiplier * training_examples[j]->lagrange_multiplier * training_examples[i]->desired_output * training_examples[j]->desired_output * kernel_function(training_examples[i]->training_point, training_examples[j]->training_point);
            }
        }

        cout << "to maximize: " << to_maximize << endl;
        */

        if (examine_all) examine_all = false;
        else if (number_changed == 0) examine_all = true;
    }

    long end_training_time = time(NULL);
    cout << "Finished training SVM in " << (end_training_time - begin_training_time) << " seconds." << endl;
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
