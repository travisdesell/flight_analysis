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

public class RankSearches {

    public static class NNInfo {
        public double best_fitness;
        public String type;
        public double rank = -1;
        public double avg_evals;

        NNInfo(String type, double best_fitness, double avg_evals) {
            this.type = type;
            this.best_fitness = best_fitness;
            this.avg_evals = avg_evals;
        }   
    }   

    public static void main(String[] arguments) {
//        Hashtable<String, Hashtable<String, Hashtable<String, ArrayList<Double>>>> output_table_fitness = new Hashtable<String, Hashtable<String, Hashtable<String, ArrayList<Double>>>>();
//        Hashtable<String, Hashtable<String, Hashtable<String, ArrayList<Integer>>>> output_table_evaluations = new Hashtable<String, Hashtable<String, Hashtable<String, ArrayList<Integer>>>>();

//        Hashtable<String, Hashtable<String, ArrayList<Double>>> nn_table_fitness = new Hashtable<String, Hashtable<String, ArrayList<Double>>>();
//        Hashtable<String, Hashtable<String, ArrayList<Integer>>> nn_table_evaluations = new Hashtable<String, Hashtable<String, ArrayList<Integer>>>();

        System.out.println("arguments.length: " + arguments.length);

        BufferedReader inputStream = null;
        String l1;

        Hashtable<String, ArrayList<Double>> network_ranks = new Hashtable<String, ArrayList<Double>>();
        Hashtable<String, Double> network_evals = new Hashtable<String, Double>();

        for (int i = 0; i < arguments.length; i++) {
            System.out.println(arguments[i]);

            ArrayList<NNInfo> networks = new ArrayList<NNInfo>();

            try {
                inputStream = new BufferedReader(new FileReader(arguments[i]));

                l1 = inputStream.readLine();    //skip the first line
                while ((l1 = inputStream.readLine()) != null) {

                    //parse the network name
                    int first_index = l1.indexOf("\"") + 1;
                    int second_index = l1.indexOf("\\n");
                    String nn_type = l1.substring(first_index, second_index);
                    
                    String[] ss = l1.split(" +");
//                    for (String s : ss) System.out.println("\t\t" + s);

                    double best_fitness = Double.parseDouble(ss[3]);
                    double avg_evals = Double.parseDouble(ss[6]);
//                    System.out.println("\t" + nn_type + " - " + best_fitness);

                    networks.add(new NNInfo(nn_type, best_fitness, avg_evals));
                }
            } catch (Exception e) {
                System.err.println("ERROR: " + e);
                e.printStackTrace();
            }

            int current_rank = 1;
            for (int j = 0; j < networks.size(); j++) {
                networks.get(j).rank = current_rank;

                if (j < networks.size() - 1 && networks.get(j + 1).best_fitness == networks.get(j).best_fitness) {
                    ;   //do nothing
                } else {
                    current_rank++;
                }
            }

            for (NNInfo nn : networks) {
//                System.out.println("\t" + nn.type + " - " + nn.rank + " - " + nn.best_fitness);

                Double avg_evals = network_evals.get(nn.type);
                if (avg_evals == null) avg_evals = 0.0;
                avg_evals += nn.avg_evals;
                network_evals.put(nn.type, avg_evals);

                ArrayList<Double> ranks = network_ranks.get(nn.type);
                if (ranks == null) ranks = new ArrayList<Double>();
                ranks.add(nn.rank);

                network_ranks.put(nn.type, ranks);
            }
        }

        System.out.println("\n");

        System.out.println("\\begin{tabular}{|l|r|r|r|}");
        System.out.println("\\hline");
        System.out.println("Network & Rank & Ranks & Avg. Evals\\\\");
        System.out.println("\\hline\\hline");

        TreeSet<String> nn_types = new TreeSet<String>(network_ranks.keySet());
        for (String nn_type : nn_types) {
            ArrayList<Double> ranks = network_ranks.get(nn_type);
            Collections.sort(ranks);

            double avg_rank = 0;
            int avg_evals = (int)network_evals.get(nn_type).doubleValue();
            for (Double rank : ranks) avg_rank += rank;
            avg_rank /= ranks.size();
            avg_evals /= ranks.size();

            System.out.print(nn_type + " & "  + avg_rank + " &");
            boolean first = true;
            for (Double rank : ranks) {
                if (first) {
                    System.out.print(" ");
                    first = false;
                } else {
                    System.out.print(", ");
                }
                System.out.print((int)rank.doubleValue());
            }
            System.out.println(" & " + avg_evals + "\\\\");
            System.out.println("\\hline");
        }
        System.out.println("\\end{tabular}");
    }
}
