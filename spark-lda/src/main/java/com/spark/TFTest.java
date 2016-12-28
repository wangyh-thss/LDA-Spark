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
        final int topicNum = Integer.parseInt(args[3]);
        SparkConf conf = new SparkConf().setAppName("datapreTest");
        JavaSparkContext sc = new JavaSparkContext(conf);

        JavaPairRDD<Long, String> source = JavaPairRDD.fromJavaRDD(sc.textFile(inputFile).zipWithIndex().map(
            new Function<Tuple2<String,Long>, Tuple2<Long, String>>() {
                public Tuple2<Long, String> call(Tuple2<String, Long> doc_id) {
                    return doc_id.swap();
                }
            }
        )).cache();

        final HashingTF hashingTF = new HashingTF();
        JavaPairRDD<Long, Vector> corpus = JavaPairRDD.fromJavaRDD(source.map(
            new Function<Tuple2<Long,String>, Tuple2<Long, Vector>>() {
                public Tuple2<Long, Vector> call(Tuple2<Long, String> t) {
                    Long tid = t._1;
                    String s = t._2;
                    String sentences = s.split("\t\n")[0].split("\2")[18];
                    String[] values = sentences.trim().split(",");
                    List<String> document = Arrays.asList(values);
                    Vector tf = hashingTF.transform(document);
                    return Tuple2.apply(tid, tf);
                }
            }
        ));

        corpus.cache();

        LDA lda = new LDA().setK(topicNum);
        DistributedLDAModel ldaModel = (DistributedLDAModel)lda.run(corpus);

        JavaPairRDD<Long, Vector> topicDist = JavaPairRDD.fromJavaRDD(ldaModel.topicDistributions().toJavaRDD().map(
                new Function<Tuple2<Object, Vector>, Tuple2<Long, Vector>>() {
                    public Tuple2<Long, Vector> call(Tuple2<Object, Vector> t) {
                        Long tid = (Long)t._1;
                        return Tuple2.apply(tid, t._2);
                    }
                }
        ));

        JavaPairRDD<Long, Tuple2<String, Vector>> docInfo = source.join(topicDist);
        JavaRDD<String> textResult = docInfo.values().map(
            new Function<Tuple2<String, Vector>, String>() {
                public String call(Tuple2<String, Vector> doc) {
                    String docText = doc._1.trim();
                    double[] topics = doc._2.toArray();
                    double avg = 1.0 / topicNum;
                    int value;

                    docText += "\2";
                    for (int i = 0; i < topics.length; i++) {
                        if (topics[i] < avg) {
                            value = 0;
                        } else {
                            value = 1;
                        }
                        docText += value;
                        if (i < topics.length - 1) {
                            docText += ",";
                        }
                    }
                    docText += "\n";
                    return docText;
                }
            }
        );
        textResult.saveAsTextFile(outputFile);
//        for (Object el: topicDistList) {
//            Tuple2 tel = (Tuple2) el;
//            String stel = tel._2().toString();
//            String[] s = stel.substring(1, stel.length() - 2).split(",");
//            for (int j = 0; j < s.length; j++) {
//                System.out.println(Double.parseDouble(s[j]));
//            }
//        }

//        JavaRDD<Tuple2<Object, Vector>> temp = topicDist.toJavaRDD();
//        for(Tuple2<Object, Vector> t: temp.collect()) {
//            System.out.println(t);
//        }
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
