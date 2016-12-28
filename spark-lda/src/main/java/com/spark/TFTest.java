package com.spark; /**
 * Created by wangyihan on 2016/12/19.
 */

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.DistributedLDAModel;
import org.apache.spark.mllib.clustering.LDA;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.rdd.RDD;
import scala.Tuple2;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import java.util.List;

public class TFTest {
    public static void main(String[] args) throws IOException {
        String inputFile = args[0];
        String outputFile = args[1];
        String modelDir = args[2];
        int topicNum = Integer.parseInt(args[3]);
        SparkConf conf = new SparkConf().setAppName("datapreTest");
        JavaSparkContext sc = new JavaSparkContext(conf);

        JavaRDD<String> aa = sc.textFile(inputFile).cache();
        JavaRDD<List<String>> documents = aa.map(
                new Function<String, List<String>>() {
                    public List<String> call(String s) {
                        String sentences = s.split("\t\n")[0].split("\2")[18];
                        String[] values = sentences.trim().split(",");
                        return Arrays.asList(values);
                    }
                }
        );

        HashingTF hashingTF = new HashingTF();
        JavaRDD<Vector> tf = hashingTF.transform(documents);
//        System.out.println(tf.collect());

        JavaPairRDD<Long, Vector> corpus = JavaPairRDD.fromJavaRDD(tf.zipWithIndex().map(
                new Function<Tuple2<Vector, Long>, Tuple2<Long, Vector>>() {
                    public Tuple2<Long, Vector> call(Tuple2<Vector, Long> doc_id) {
                        return doc_id.swap();
                    }
                }
        ));
        corpus.cache();

//        int topicNum = 10;
        LDA lda = new LDA().setK(topicNum);
        DistributedLDAModel ldaModel = (DistributedLDAModel)lda.run(corpus);

        RDD<Tuple2<Object, Vector>> topicDist = ldaModel.topicDistributions();
        Tuple2[] topicDistList = (Tuple2[])topicDist.collect();

        for (Object el: topicDistList) {
            Tuple2 tel = (Tuple2) el;
            String stel = tel._2().toString();
            String[] s = stel.substring(1, stel.length() - 2).split(",");
            for (int j = 0; j < s.length; j++) {
                System.out.println(Double.parseDouble(s[j]));
            }
        }

        JavaRDD<Tuple2<Object, Vector>> temp = topicDist.toJavaRDD();
        for(Tuple2<Object, Vector> t: temp.collect()) {
            System.out.println(t);
        }
//        temp.saveAsTextFile(outputFile);

//        JavaRDD<String> topicDistList = JavaRDD.fromRDD(topicDist, topicDist.elementClassTag()).map(
//                new Function<Tuple2<Object, Vector>, String>() {
//                    public String call(Tuple2<Object, Vector> topic) {
//                        System.out.println(topic._1());
//                        System.out.println(topic._2());
//                        return "";
//                    }
//                }
//        );




//        System.out.println("Learned topics (as distributions over vocab of " + ldaModel.vocabSize() + " words):");
//        Matrix topics = ldaModel.topicsMatrix();
//        for (int topic = 0; topic < 3; topic++) {
//            System.out.print("Topic " + topic + ":");
//            for (int word = 0; word < ldaModel.vocabSize(); word++) {
////                System.out.print(" " + topics.apply(word, topic));
//            }
//            System.out.println();
//        }

        // Save LDA Model
        new TFTest().save(lda, ldaModel, modelDir);
        sc.stop();
    }

    public void save(LDA lda, DistributedLDAModel ldaModel, String outputDir) throws IOException {
        LDAPredict ldaPredict = new LDAPredict(lda, ldaModel.toLocal());
        ObjectOutputStream objout = new ObjectOutputStream(new FileOutputStream(
                new File(outputDir, "ldaPredict.mod")));
        objout.writeObject(ldaPredict);
    }
}
