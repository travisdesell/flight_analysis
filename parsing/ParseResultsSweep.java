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

public class ParseResultsSweep {

    public static class Pair {
        public double min_fitness;
        public String output;

        Pair(double min_fitness, String output) {
            this.min_fitness = min_fitness;
            this.output = output;
        }
    }

    public static void main(String[] arguments) {
        Hashtable<String, Hashtable<String, Hashtable<String, ArrayList<Double>>>> output_table_fitness = new Hashtable<String, Hashtable<String, Hashtable<String, ArrayList<Double>>>>();
        Hashtable<String, Hashtable<String, Hashtable<String, ArrayList<Integer>>>> output_table_evaluations = new Hashtable<String, Hashtable<String, Hashtable<String, ArrayList<Integer>>>>();

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

                search_type = search_type.replace("random", "rand");
                search_type = search_type.replace("binary", "bin");

                l3 = inputStream.readLine();

                l4 = inputStream.readLine();
                double fitness = Double.parseDouble(l4) * -1.0;

                l5 = inputStream.readLine();

                Hashtable<String, Hashtable<String, ArrayList<Double>>> nn_table_fitness = output_table_fitness.get(output_target);
                Hashtable<String, Hashtable<String, ArrayList<Integer>>> nn_table_evaluations = output_table_evaluations.get(output_target);

                if (nn_table_evaluations == null) nn_table_evaluations = new Hashtable<String, Hashtable<String, ArrayList<Integer>>>();
                if (nn_table_fitness == null) nn_table_fitness = new Hashtable<String, Hashtable<String, ArrayList<Double>>>();

//                System.out.println(String.format("%-75s :     %-8d : %-1.10f", type, evaluations, fitness));

                Hashtable<String, ArrayList<Integer>> search_table_evaluations = nn_table_evaluations.get(nn_type);
                Hashtable<String, ArrayList<Double>> search_table_fitness = nn_table_fitness.get(nn_type);

                if (search_table_evaluations == null) search_table_evaluations = new Hashtable<String, ArrayList<Integer>>();
                if (search_table_fitness == null) search_table_fitness = new Hashtable<String, ArrayList<Double>>();

                ArrayList<Integer> evaluations_list = search_table_evaluations.get(search_type);
                ArrayList<Double> fitness_list = search_table_fitness.get(search_type);

                if (evaluations_list == null) evaluations_list = new ArrayList<Integer>();
                if (fitness_list == null) fitness_list = new ArrayList<Double>();

                evaluations_list.add(evaluations);
                fitness_list.add(fitness);

                search_table_evaluations.put(search_type, evaluations_list);
                search_table_fitness.put(search_type, fitness_list);

                nn_table_evaluations.put(nn_type, search_table_evaluations);
                nn_table_fitness.put(nn_type, search_table_fitness);

                output_table_evaluations.put(output_target, nn_table_evaluations);
                output_table_fitness.put(output_target, nn_table_fitness);
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

            TreeSet<String> nn_types = new TreeSet<String>(nn_table_fitness.keySet());
            for (String nn_type : nn_types) {
                Hashtable<String, ArrayList<Double>> search_table_fitness = nn_table_fitness.get(nn_type);
                Hashtable<String, ArrayList<Integer>> search_table_evaluations = nn_table_evaluations.get(nn_type);

                System.out.println("    " + nn_type);

                TreeSet<String> search_types = new TreeSet<String>( search_table_fitness.keySet() );
                for (String search_type : search_types) {
                    ArrayList<Double> fitness_list = search_table_fitness.get(search_type);
                    ArrayList<Integer> evaluations_list = search_table_evaluations.get(search_type);

                    double min_fitness = Double.MAX_VALUE, max_fitness = Double.MIN_VALUE, avg_fitness = 0;
                    for (double f : fitness_list) {
                        if (f < min_fitness) min_fitness = f;
                        if (f > max_fitness) max_fitness = f;
                        avg_fitness += f;
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
                    String output2 = String.format("              %-50s   %15.10f   %15.10f   %15.10f   %-10d   %-10d   %-10d   %-10d", "\"" + search_type + "\"", min_fitness, avg_fitness, max_fitness, min_evaluations, avg_evaluations, max_evaluations, fitness_list.size());

                    System.out.println(output2);

                    
                    Pair p = new Pair(min_fitness, output);
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

        for (String output_target : output_targets) {
            System.out.println("\nSORTED " + output_target + ":");
            System.out.println("\"Network Type - Search Type\"                                    \"min fitness\"     \"avg fitness\"     \"max fitness\"  \"min evals\"  \"avg evals\"  \"max evals\"  runs");

            ArrayList<Pair> pairs = pairs_table.get(output_target);
            Collections.sort(pairs, new Comparator<Pair>() {
                public int compare(Pair p1, Pair p2) {
                    return  new Double(p1.min_fitness).compareTo(new Double(p2.min_fitness));
                }
            });

            for (Pair p : pairs) {
                System.out.println("    " + p.output);
            }
        }

        System.out.println("total runs: " + total_runs);
    }
}
