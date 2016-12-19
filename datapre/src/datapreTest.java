import groovy.lang.MetaClassImpl;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import scala.Tuple2;
import scala.tools.cmd.gen.AnyVals;

import java.util.*;

/**
 * Created by xuziru on 2016/12/14.
 */

public class datapreTest {
    public static void main(String[] args) {
        String inputFile = "data/testdata";
        String outputDic = "data/dic";
        SparkConf conf = new SparkConf().setAppName("datapreTest");
        JavaSparkContext sc = new JavaSparkContext(conf);

        JavaRDD<String> textFile = sc.textFile(inputFile);
        long documentCount = textFile.count();

        // get dictionary
        // JavaRDD<String> documents = textFile.map(new Function<String, String>() {
        //     @Override
        //     public String call(String s) throws Exception {
        //         String sentences = s.split("\t\n")[0].split("\2")[18];
        //         return sentences;
        //     }
        // });

        final JavaRDD<String> words = textFile.flatMap(new FlatMapFunction<String, String>() {
            @Override
            public Iterable<String> call(String s) throws Exception {
            String sentences = s.split("\t\n")[0].split("\2")[18];
            return Arrays.asList(sentences.split(","));
            }
        });
        JavaPairRDD<String, Integer> pairs = words.mapToPair(new PairFunction<String, String, Integer>() {
            @Override
            public Tuple2<String, Integer> call(String s) throws Exception {
            return new Tuple2<String, Integer>(s, 1);
            }
        });
        JavaPairRDD<String, Integer> counts = pairs.reduceByKey(new Function2<Integer, Integer, Integer>() {
            @Override
            public Integer call(Integer a, Integer b) throws Exception {
                return a + b;
            }
        });
        counts.saveAsTextFile(outputDic);

        Map<String, Integer> countsMap = counts.collectAsMap();
        Set<String> keySet = countsMap.keySet();
        int wordsCount = keySet.size();
        System.out.println(keySet);
        System.out.println(wordsCount);
        System.out.println("output dictionary finish");



        sc.close();
    }
}
