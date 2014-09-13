import java.io.FileReader;
import java.io.FileWriter;
import java.io.BufferedReader;
import java.io.PrintWriter;
import java.io.IOException;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Collections;
import java.util.Hashtable;
import java.util.TreeSet;

public class ParseResults {

    public static class Pair {
        public double min_fitness;
        public String output;
        public String nn_type;
        public String network;

        Pair(double min_fitness, String output, String nn_type, String network) {
            this.min_fitness = min_fitness;
            this.output = output;
            this.nn_type = nn_type;
            this.network = network;
        }
    }

    public static void main(String[] arguments) {
        int flight_id = Integer.parseInt(arguments[1]);

        Hashtable<String, Hashtable<String, Hashtable<String, ArrayList<Integer>>>> output_table_evaluations = new Hashtable<String, Hashtable<String, Hashtable<String, ArrayList<Integer>>>>();
        Hashtable<String, Hashtable<String, Hashtable<String, ArrayList<Double>>>> output_table_fitness = new Hashtable<String, Hashtable<String, Hashtable<String, ArrayList<Double>>>>();
        Hashtable<String, Hashtable<String, Hashtable<String, ArrayList<String>>>> output_table_networks = new Hashtable<String, Hashtable<String, Hashtable<String, ArrayList<String>>>>();

//        Hashtable<String, Hashtable<String, ArrayList<Double>>> nn_table_fitness = new Hashtable<String, Hashtable<String, ArrayList<Double>>>();
//        Hashtable<String, Hashtable<String, ArrayList<Integer>>> nn_table_evaluations = new Hashtable<String, Hashtable<String, ArrayList<Integer>>>();

        BufferedReader inputStream = null;

        String l1 = null, l2 = null, l3 = null, l4 = null, l5 = null;
        try {
            inputStream = new BufferedReader(new FileReader(arguments[0]));

            while ((l1 = inputStream.readLine()) != null) {
                l1 = l1.substring(l1.indexOf("out_"), l1.length());
                String type = l1.substring(0, l1.indexOf("__run"));

                l2 = inputStream.readLine();
                int iterations = new Integer(l2);

                int population_size = Integer.parseInt(l1.substring(l1.lastIndexOf("_p") + 2, l1.indexOf("__", l1.lastIndexOf("_p"))));
                //System.out.println("population size: " + population_size);
                int evaluations = iterations * population_size;
                String search_type = l1.substring(l1.indexOf("__") + 2, l1.indexOf("__", l1.indexOf("__") + 2));
                String nn_type = l1.substring(l1.indexOf("__", l1.indexOf("__") + 2) + 2, l1.indexOf("__run"));

                String output_target = "";
                try {
                    int first_pos = l1.indexOf("_", l1.indexOf("_") + 1) + 1;
                    output_target = l1.substring(first_pos, l1.indexOf("__"));
                } catch (Exception e) {
                    output_target = "";
                }

                nn_type = nn_type.replace("_b","");

                search_type = search_type.replace("random", "rand");
                search_type = search_type.replace("binary", "bin");

                l3 = inputStream.readLine();
                String network_weights = l3;

//                System.out.println("network weights: " + network_weights);

                l4 = inputStream.readLine();
                double fitness = Double.parseDouble(l4) * -1.0;

                l5 = inputStream.readLine();

                Hashtable<String, Hashtable<String, ArrayList<Integer>>> nn_table_evaluations = output_table_evaluations.get(output_target);
                Hashtable<String, Hashtable<String, ArrayList<Double>>> nn_table_fitness = output_table_fitness.get(output_target);
                Hashtable<String, Hashtable<String, ArrayList<String>>> nn_table_networks  = output_table_networks.get(output_target);

                if (nn_table_evaluations == null) nn_table_evaluations = new Hashtable<String, Hashtable<String, ArrayList<Integer>>>();
                if (nn_table_fitness == null) nn_table_fitness = new Hashtable<String, Hashtable<String, ArrayList<Double>>>();
                if (nn_table_networks == null) nn_table_networks = new Hashtable<String, Hashtable<String, ArrayList<String>>>();

//                System.out.println(String.format("%-75s :     %-8d : %-1.10f", type, evaluations, fitness));

                Hashtable<String, ArrayList<Integer>> search_table_evaluations = nn_table_evaluations.get(nn_type);
                Hashtable<String, ArrayList<Double>> search_table_fitness = nn_table_fitness.get(nn_type);
                Hashtable<String, ArrayList<String>> search_table_networks = nn_table_networks.get(nn_type);

                if (search_table_evaluations == null) search_table_evaluations = new Hashtable<String, ArrayList<Integer>>();
                if (search_table_fitness == null) search_table_fitness = new Hashtable<String, ArrayList<Double>>();
                if (search_table_networks == null) search_table_networks = new Hashtable<String, ArrayList<String>>();

                ArrayList<Integer> evaluations_list = search_table_evaluations.get(search_type);
                ArrayList<Double> fitness_list = search_table_fitness.get(search_type);
                ArrayList<String> networks_list = search_table_networks.get(search_type);

                if (evaluations_list == null) evaluations_list = new ArrayList<Integer>();
                if (fitness_list == null) fitness_list = new ArrayList<Double>();
                if (networks_list == null) networks_list = new ArrayList<String>();

                evaluations_list.add(evaluations);
                fitness_list.add(fitness);
                networks_list.add(network_weights);

                search_table_evaluations.put(search_type, evaluations_list);
                search_table_fitness.put(search_type, fitness_list);
                search_table_networks.put(search_type, networks_list);

                nn_table_evaluations.put(nn_type, search_table_evaluations);
                nn_table_fitness.put(nn_type, search_table_fitness);
                nn_table_networks.put(nn_type, search_table_networks);

                output_table_evaluations.put(output_target, nn_table_evaluations);
                output_table_fitness.put(output_target, nn_table_fitness);
                output_table_networks.put(output_target, nn_table_networks);
            }

            inputStream.close();
        } catch(Exception e) {
            System.err.println("Exception occured parsing file: " + e);
            System.err.println("l1: " + l1);
            System.err.println("l2: " + l2);
            System.err.println("l3: " + l3);
            System.err.println("l4: " + l4);
            System.err.println("l5: " + l5);
            e.printStackTrace();
        }

        //System.out.println("\n\n");

        double best_fitness = Double.MAX_VALUE;
        String best = null;

        Hashtable<String, ArrayList<Pair>> pairs_table = new Hashtable<String, ArrayList<Pair>>();

        int total_runs = 0;
        System.out.format("%-60s     : %15s : %15s : %15s : %-10s : %-10s : %-10s : %-10s\n", "nn_search", "min_fit", "avg_fit", "max_fit", "min_evals", "avg_evals", "max_evals", "count");

        TreeSet<String> output_targets = new TreeSet<String>(output_table_fitness.keySet());
        for (String output_target : output_targets) {
            System.out.println(output_target + ":");

            Hashtable<String, Hashtable<String, ArrayList<Double>>> nn_table_fitness = output_table_fitness.get(output_target);
            Hashtable<String, Hashtable<String, ArrayList<Integer>>> nn_table_evaluations = output_table_evaluations.get(output_target);
            Hashtable<String, Hashtable<String, ArrayList<String>>> nn_table_networks = output_table_networks.get(output_target);

            TreeSet<String> nn_types = new TreeSet<String>(nn_table_fitness.keySet());
            for (String nn_type : nn_types) {
                Hashtable<String, ArrayList<Double>> search_table_fitness = nn_table_fitness.get(nn_type);
                Hashtable<String, ArrayList<Integer>> search_table_evaluations = nn_table_evaluations.get(nn_type);
                Hashtable<String, ArrayList<String>> search_table_networks = nn_table_networks.get(nn_type);

                System.out.println("    " + nn_type);

                TreeSet<String> search_types = new TreeSet<String>( search_table_fitness.keySet() );
                for (String search_type : search_types) {
                    ArrayList<Double> fitness_list = search_table_fitness.get(search_type);
                    ArrayList<Integer> evaluations_list = search_table_evaluations.get(search_type);
                    ArrayList<String> networks_list = search_table_networks.get(search_type);

                    int best_fitness_index = 0;
                    int index = 0;
                    double min_fitness = Double.MAX_VALUE, max_fitness = Double.MIN_VALUE, avg_fitness = 0;
                    for (double f : fitness_list) {
                        if (f < min_fitness) {
                            min_fitness = f;
                            best_fitness_index = index;
                        }
                        if (f > max_fitness) max_fitness = f;
                        avg_fitness += f;
                        index++;
                    }
                    avg_fitness /= fitness_list.size();

                    int min_evaluations = Integer.MAX_VALUE, max_evaluations = Integer.MIN_VALUE, avg_evaluations = 0;
                    for (int e : evaluations_list) {
                        if (e < min_evaluations) min_evaluations = e;
                        if (e > max_evaluations) max_evaluations = e;
                        avg_evaluations += e;
                    }
                    avg_evaluations /= evaluations_list.size();

                    //String output = String.format("    %-60s : %15.10f : %15.10f : %15.10f : %-10d : %-10d : %-10d : %-10d", output_target + " - " + nn_type + " - " + search_type, min_fitness, avg_fitness, max_fitness, min_evaluations, avg_evaluations, max_evaluations, fitness_list.size());
                    search_type = search_type.replace("_", "/");
                    nn_type = nn_type.replace("_", "/");
                    nn_type = nn_type.replace("feed/forward", "feed forward");
                    String output = String.format("    %-58s   %15.10f   %15.10f   %15.10f   %-10d   %-10d   %-10d   %-10d", "\"" + nn_type + "\\n" + search_type + "\"", min_fitness, avg_fitness, max_fitness, min_evaluations, avg_evaluations, max_evaluations, fitness_list.size());
                    String output2 = String.format("              %-50s : %15.10f : %15.10f : %15.10f : %-10d : %-10d : %-10d : %-10d", search_type, min_fitness, avg_fitness, max_fitness, min_evaluations, avg_evaluations, max_evaluations, fitness_list.size());

                    System.out.println(output2);

                    
                    Pair p = new Pair(min_fitness, output, nn_type, networks_list.get(best_fitness_index));
                    ArrayList<Pair> pairs = pairs_table.get(output_target);
                    if (pairs == null) pairs = new ArrayList<Pair>();
                    pairs.add(p);
                    pairs_table.put(output_target, pairs);

                    if (min_fitness < best_fitness) {
                        best_fitness = min_fitness;
                        best = output;
                    }
                    total_runs += fitness_list.size();
                }
            }
        }

        ArrayList<Pair> best_pairs = new ArrayList<Pair>();

        for (String output_target : output_targets) {
            try {
            System.out.println("\nSORTED " + output_target + ":");

            PrintWriter writer = new PrintWriter(flight_id + "_" + output_target + ".txt", "UTF-8");
            System.out.println("\"Network Type - Search Type\"                                    \"min fitness\"     \"avg fitness\"     \"max fitness\"  \"min evals\"  \"avg evals\"  \"max evals\"  runs");
            writer.println("\"Network Type - Search Type\"                                    \"min fitness\"     \"avg fitness\"     \"max fitness\"  \"min evals\"  \"avg evals\"  \"max evals\"  runs");

            ArrayList<Pair> pairs = pairs_table.get(output_target);
            Collections.sort(pairs, new Comparator<Pair>() {
                public int compare(Pair p1, Pair p2) {
                    return  new Double(p1.min_fitness).compareTo(new Double(p2.min_fitness));
                }
            });

            for (Pair p : pairs) {
                System.out.println("    " + p.output);
                writer.println("    " + p.output);
                
            }
            writer.close();

            best_pairs.add(new Pair(pairs.get(0).min_fitness, output_target, pairs.get(0).nn_type, pairs.get(0).network));

            } catch (Exception e) {
                System.err.println("Error writing to file: '" + flight_id + "_" + output_target + ".txt'");
                System.err.println(e);
                e.printStackTrace();
            }
        }

        System.out.println("best networks: ");
        for (Pair p : best_pairs) {
            System.out.println(flight_id + " : " + p.output + " : " + p.nn_type + " : " + p.min_fitness + " : " + p.network);
        }

        System.out.println("total runs: " + total_runs);
    }
}
