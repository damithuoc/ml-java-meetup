/*
 * Copyright (c) 2015, WSO2 Inc. (http://www.wso2.org) All Rights Reserved.
 *
 * WSO2 Inc. licenses this file to you under the Apache License,
 * Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package com.nirmal.ml;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

import scala.Tuple2;

/**
 * Spark logistic regression demo.
 */
public class SparkLogisticRegresser {

    public static void main(String[] args) throws InterruptedException {

        // 1. initialize spark conf
        SparkConf sparkConf = new SparkConf();
        // your application name
        sparkConf.setAppName("ML-JAVA-MEETUP");
        // spark is running on embedded mode in the same JVM as this app. If you are connecting to a external Spark
        // cluster, you have to mention the Spark master URL here.
        sparkConf.setMaster("local");

        // 2. initialize Java spark context
        JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);

        // 3. read the dataset from file system
        /*
         * Lines RDD would be something like;
         * "f1","f2","f3"...
         * "1","9","8"...
         * "3","2","4"...
         * "6","0","0"...
         * ....
         * ....
         */
        JavaRDD<String> lines = sparkContext.textFile("file:/Users/nirmal/Desktop/dataset/IndiansDiabetes.csv");
        // reads the first line which is the header row.
        final String headerRow = lines.first();
        System.out.println("Header row: " + headerRow);

        // 4. filter out the header row from the dataset
        /*
         * Filtered RDD would be something like;
         * "1","9","8"...
         * "3","2","4"...
         * "6","0","0"...
         * ....
         * ....
         */
        JavaRDD<String> linesWithoutHeaderRow = lines.filter(new Function<String, Boolean>() {

            private static final long serialVersionUID = -7442877753878620445L;

            @Override
            public Boolean call(String arg0) throws Exception {
                if (headerRow.equals(arg0)) {
                    // return false, since this row shouldn't be kept.
                    return false;
                }
                return true;
            }
        });

        // 4. process row wise and split into tokens 
        /*
         * New RDD would be something like;
         * ["1","9","8"...]
         * ["3","2","4"...]
         * ["6","0","0"...]
         * ....
         * ....
         */
        JavaRDD<String[]> tokens = linesWithoutHeaderRow.map(new Function<String, String[]>() {

            private static final long serialVersionUID = 1035500695508821504L;

            @Override
            public String[] call(String arg0) throws Exception {
                System.out.println(arg0);
                return arg0.split(",");
            }

        });

        // 5. process row wise and exclude features
        /*
         * New RDD would be something like;
         * ["1","8","1"...]
         * ["3","4","2"...]
         * ["6","0","5"...]
         * ....
         * ....
         */
        final int featureIdxToBeExcluded = 1;
        JavaRDD<String[]> tokensAfterExcludingFeatures = tokens.map(new Function<String[], String[]>() {

            private static final long serialVersionUID = 8278130586575522865L;

            @Override
            public String[] call(String[] arg0) throws Exception {
                String[] newRow = new String[arg0.length - 1];
                int j = 0;
                for (int i = 0; i < arg0.length; i++) {
                    // excluding feature
                    if (i != featureIdxToBeExcluded) {
                        newRow[j++] = arg0[i];
                    }
                }
                System.out.println("Original feature count: " + arg0.length + " --- New feature count: "
                        + newRow.length);
                return newRow;
            }

        });

        // convert strings to doubles
        /*
         * New RDD would be something like;
         * [1.0,8.0,1.0...]
         * [3.0,4.0,2.0...]
         * [6.0,0.0,5.0...]
         * ....
         * ....
         */
        JavaRDD<double[]> tokensAsDoubles = tokensAfterExcludingFeatures.map(new Function<String[], double[]>() {

            private static final long serialVersionUID = -8508389002212459624L;

            @Override
            public double[] call(String[] arg0) throws Exception {
                double[] features = new double[arg0.length];
                for (int i = 0; i < arg0.length; ++i) {
                    features[i] = Double.parseDouble(arg0[i]);
                }
                return features;
            }
        });

        // convert doubles to LabeledPoint
        // LabeledPoint = response variable and feature vector
        /*
         * Input RDD;
         * [1.0,8.0,1.0...,0]
         * [3.0,4.0,2.0...,1]
         * [6.0,0.0,5.0...,1]
         * ....
         * ....
         * 
         * Resulted RDD would be something like (if printed);
         * [1.0,8.0,1.0...,0.7][0]
         * [3.0,4.0,2.0...,2.1][1]
         * [6.0,0.0,5.0...,1.1][1]
         * ....
         * ....
         * 
         */
        JavaRDD<LabeledPoint> tokensAsLabeledPoints = tokensAsDoubles.map(new Function<double[], LabeledPoint>() {

            private static final long serialVersionUID = 9108848373390703605L;

            @Override
            public LabeledPoint call(double[] arg0) throws Exception {
                // last index is the response value after the upstream transformations
                double response = arg0[arg0.length - 1];
                // new feature vector does not contain response variable value
                double[] features = new double[arg0.length - 1];
                for (int i = 0; i < arg0.length - 1; i++) {
                    features[i] = arg0[i];
                }
                return new LabeledPoint(response, Vectors.dense(features));
            }

        });

        // split the dataset into two portions - training dataset and testing dataset
        JavaRDD<LabeledPoint>[] preprocessedData = tokensAsLabeledPoints.randomSplit(new double[] { 0.7, 0.3 });

        // first split is the training dataset and let's cache it (to speedup the training)
        JavaRDD<LabeledPoint> trainingDataset = preprocessedData[0].cache();
        JavaRDD<LabeledPoint> testDataset = preprocessedData[1];

        // train the logistic regression model
        LogisticRegressionWithSGD lrSGD = new LogisticRegressionWithSGD();
        final LogisticRegressionModel model = lrSGD.run(trainingDataset.rdd());
        System.out.println("Threshold : " + model.getThreshold().get());
        // clear the threshold so that we can get a probability as the prediction
        model.clearThreshold();

        // print the model as a PMML (predictive model markup language) so that we can see the coefficients (weights)
        System.out.println(model.toPMML());

        // training dataset is no more of use, let's get it out from memory
        trainingDataset.unpersist();
        // let's cache the test dataset.
        testDataset.cache();
        
        // Use the built model to predict the output for the test dataset and get a RDD with the predicted value and corresponding real value.
        /*
         * This RDD would be something like;
         * 0.76,1
         * 0.12,0
         * 0.45,1
         * ....
         * ....
         */
        JavaRDD<Tuple2<Object, Object>> scoresAndLabels = testDataset
                .map(new Function<LabeledPoint, Tuple2<Object, Object>>() {
                    private static final long serialVersionUID = 910861043765821669L;

                    public Tuple2<Object, Object> call(LabeledPoint labeledPoint) {
                        Double score = model.predict(labeledPoint.features());
                        return new Tuple2<Object, Object>(score, labeledPoint.label());
                    }
                });

        System.out.println(scoresAndLabels.collect());

        // Use Spark's MulticlassMetrics to obtain the performance figures of this evaluation
        MulticlassMetrics metrics = new MulticlassMetrics(scoresAndLabels.rdd());
        System.out.println(metrics.confusionMatrix().toString());
        System.out.println("Precision : " + metrics.precision());
        System.out.println("Recall : " + metrics.recall());

        // take test dataset out from the memory
        testDataset.unpersist();

        // to show the Spark UI
        // Thread.sleep(120000);

        // at the end of the task, close the SparkContext
        sparkContext.close();
    }

}
