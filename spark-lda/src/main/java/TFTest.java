/**
 * Created by wangyihan on 2016/12/19.
 */

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.DistributedLDAModel;
import org.apache.spark.mllib.clustering.LDA;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.feature.HashingTF;
import scala.Tuple2;

import java.util.Arrays;
import java.util.List;

public class TFTest {
    public static void main(String[] args) {
        String inputFile = "data/testdata";
        String outputDic = "data/dic";
        SparkConf conf = new SparkConf().setAppName("datapreTest");
        JavaSparkContext sc = new JavaSparkContext(conf);


        JavaRDD<List<String>> documents = sc.textFile(inputFile).map(
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

        DistributedLDAModel ldaModel = (DistributedLDAModel)new LDA().setK(3).run(corpus);

        System.out.println("Learned topics (as distributions over vocab of " + ldaModel.vocabSize() + " words):");
        Matrix topics = ldaModel.topicsMatrix();
        for (int topic = 0; topic < 3; topic++) {
            System.out.print("Topic " + topic + ":");
            for (int word = 0; word < ldaModel.vocabSize(); word++) {
                System.out.print(" " + topics.apply(word, topic));
            }
            System.out.println();
        }
        sc.stop();
    }
}
