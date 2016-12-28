package java.com.spark; /**
 * Created by wangyihan on 2016/12/20.
 */

import java.io.*;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.DistributedLDAModel;
import org.apache.spark.mllib.clustering.LocalLDAModel;
import org.apache.spark.mllib.clustering.LDA;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.util.SystemClock;
import scala.Tuple2;
import scala.tools.nsc.backend.icode.Members;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class LDAPredict implements Serializable {
    LocalLDAModel ldaModel = null;
    int numTopics = 0;
    int numTerms = 0;
    Matrix topicTermsMatrix = null;
    double alpha = 0.0;

    public LDAPredict() {

    }

    public LDAPredict(LDA lda, LocalLDAModel ldaModel) {
        this.ldaModel = ldaModel;
        this.numTopics = ldaModel.k();
        this.numTerms = ldaModel.vocabSize();
        this.topicTermsMatrix = ldaModel.topicsMatrix();
        this.alpha = lda.getAlpha();
    }

    public double digamma(double x)
    {
        double p;
        x = x + 6;
        p = 1 / ( x * x );
        p = (((0.004166666666667 * p - 0.003968253986254 ) * p + 0.008333333333333) * p - 0.083333333333333) * p ;
        p = p + Math.log(x) - 0.5 / x - 1 / ( x - 1 ) - 1 / ( x - 2 ) - 1 / ( x - 3 )
                - 1 / ( x - 4 ) - 1 / ( x - 5 ) - 1 / ( x - 6 );
        return p;
    }

    public double log_sum( double log_a, double log_b )
    {
        double v;
        if (log_a < log_b)
            v = log_b + Math.log( 1 + Math.exp( log_a - log_b ) );
        else
            v = log_a + Math.log( 1 + Math.exp( log_b - log_a ) );
        return v;
    }

    public double logGamma(double x)
    {
        double z = 1 / (x * x);
        x=x+6;
        z=(((-0.000595238095238*z+0.000793650793651)*z-0.002777777777778)*z+0.083333333333333)/x;
        z=(x-0.5)*Math.log(x)-x+0.918938533204673+z-Math.log(x-1)
                -Math.log(x-2)-Math.log(x-3)-Math.log(x-4)-Math.log(x-5)-Math.log(x-6);
        return z;
    }

    public double computeLikelihood(Vector usr, double[] varGamma, double[][] phi) {
        double likelihood = 0.0;
        double digsum = 0.0;
        double varGammaSum = 0.0;
        double[] dig = new double[numTopics];
        int k, n;

        for (k = 0; k < numTopics; k++) {
            dig[k] = digamma(varGamma[k]);
            varGammaSum += varGamma[k];
        }
        digsum = digamma(varGammaSum);

        likelihood = logGamma(alpha * numTopics - numTopics * logGamma(alpha) - (logGamma(varGammaSum)));
        assert likelihood != Double.NaN;
        for (k = 0; k < numTopics; k++) {
            likelihood += (alpha - 1)*(dig[k] - digsum) + logGamma(varGamma[k])
                    - (varGamma[k] - 1) * (dig[k] - digsum);
            for (n = 0; n < numTerms; n++) {
                if (topicTermsMatrix.apply(n, k) > 0) {
                    if (phi[n][k] > 0) {
                        likelihood += usr.apply(n) * ( phi[n][k] * ( (dig[k] - digsum)
                                - Math.log(phi[n][k]) + Math.log(topicTermsMatrix.apply(n, k))));
                        assert likelihood != Double.NaN;
                    }
                }
            }
        }
        return likelihood;
    }

    public double safeLog(double x) {
        if (x == 0) {
            return 0;
        } else {
            return Math.log(x);
        }
    }

    public double[] predict(Vector usr) {
        double[] varGamma = new double[numTopics];
        double converged = 1.0;
        double phisum = 0.0;
        double likelihood = 0.0;
        double oldLikelihood = 0.0;
        double[][] phi = new double[numTerms][numTopics];
        double[] oldPhi = new double[numTopics];
        double[] digGammaTemp = new double[numTopics];
        Matrix topicTermsMatrix = this.ldaModel.topicsMatrix();

        for(int k = 0; k < numTopics; k++) {
            varGamma[k] = 1.0 / numTopics;
            digGammaTemp[k] = digamma( varGamma[k] );
            for(int n = 0; n < numTerms; n++)
                phi[n][k] = 1.0 / numTopics;
        }
        int varIter = 0;
        while(converged > 1e-6) {
            varIter += 1;
            for(int n = 0; n < numTerms; n++) {
                phisum = 0;
                for(int k = 0; k < numTopics; k++) {
                    oldPhi[k] = phi[n][k];
                    if (topicTermsMatrix.apply(n, k) > 0) {
                        phi[n][k] = digGammaTemp[k]
                                + safeLog(topicTermsMatrix.apply(n, k));
                    } else {
                        phi[n][k] = digGammaTemp[k] - 100;
                    }
                    if(k > 0) {
                        phisum = log_sum(phisum, phi[n][k]);
                    } else {
                        phisum = phi[n][k];
                    }
                }
                for(int k = 0; k < numTopics; k++) {
                    phi[n][k] = Math.exp( phi[n][k] - phisum );
                    varGamma[k] = varGamma[k] + usr.apply(n) * ( phi[n][k] - oldPhi[k] );
                    digGammaTemp[k] = digamma( varGamma[k] );
                }
            }
            likelihood = computeLikelihood(usr, varGamma, phi);
            if (1 != varIter) {
                converged = ( oldLikelihood - likelihood ) / oldLikelihood;
                oldLikelihood = likelihood;
            }
        }
        return varGamma;
    }

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        String inputFile = "data/testdata";
        SparkConf conf = new SparkConf().setAppName("ldaPredict");
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
        System.out.println("Calculate TF completed");

        List<Vector> docList = tf.collect();
        ObjectInputStream objin = new ObjectInputStream(new FileInputStream("data/model/ldaPredict.mod"));
        LDAPredict ldaPredict = (LDAPredict)objin.readObject();
        System.out.println("Load predict model compelted");
        Vector usr = docList.get(0);
        double[] result = ldaPredict.predict(usr);
        for (int i = 0; i < result.length; i++) {
            System.out.print(result[i] + ", ");
        }
        sc.stop();
    }

}
