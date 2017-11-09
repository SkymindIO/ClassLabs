package ai.skymind.training.labs;

import org.apache.log4j.BasicConfigurator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

//import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;

/**
 * EXERCISE 4: train a LSTM to predict mortality using the Physionet
 * Challenge 2012 data publicly available at https://physionet.org/challenge/2012/
 *
 *
 */
public class PhysionetMultivariateTimeSeriesClassificationBasic {

    // Change directory


    private static File baseDir = new File("src/main/resources/physionet2012");
    private static File featuresDir = new File(baseDir, "sequence");

    /* Task-specific configuration */
    private static File labelsDir = new File(baseDir, "mortality");

    // For logging with SL4J
    private static final Logger log = LoggerFactory.getLogger(PhysionetMultivariateTimeSeriesClassificationBasic.class);

    // Number of training, validation, test examples
    public static int NB_INPUTS = 84;
    public static final int NB_TRAIN_EXAMPLES = 3200;
    public static final int NB_VALID_EXAMPLES = 400;
    public static final int NB_TEST_EXAMPLES = 400;

    /**
     * ######## LAB STEP 4 #############
     * Set the Number of Epochs to 25
     *
     **/

       //######## YOUR CODE HERE ########


    /**
     * ######## LAB STEP 5 #############
     * Set the Random Seed, LearningRat, BatchSize and LSTM layer size
     *
     *
     **/

     //######## YOUR CODE HERE ########




    public static void main(String[] args) throws IOException, InterruptedException {
        BasicConfigurator.configure();



        int numLabelClasses = 2;
        //boolean resampled = ; // If true use resampled data

    /**
     * ######## LAB STEP 6 #############
     * Create DataSetIterator for trainData, validData and testData
     *
     *
     **/

        //######## YOUR CODE HERE ########




        // For this example we are using the resampled dataset
        // one set of values per hour
        // The solutions project has an example of reading the sequence data with or without the
        // inital time and offset fields.
        featuresDir = new File(baseDir, "resampled");


        /**
         * ######## LAB STEP 7 #############
         * Add SequenceRecordReaders for Labels and Features
         * Combine the Features and Labels into a DataSet
         *
         **/

        //######## YOUR CODE HERE ########



        /**
         * ######## LAB STEP 8 #############
         * Add a Neural Network Configuration
         * using a Computation Graph Configuration
         *
         **/

                //######## YOUR CODE HERE ########











        // STEP 10 uncomment this section and run the code
        // STEP #10 REMOVE THE COMMENT BELOW
        /*


        for( int i=0; i<NB_EPOCHS; i++ ){

            model.fit(trainData); // implicit inner loop over minibatches

            // loop over batches in training data to compute training AUC
            ROC roc = new ROC(100);
            trainData.reset();

            while(trainData.hasNext()){
                DataSet batch = trainData.next();
                INDArray[] output = model.output(batch.getFeatures());
                roc.evalTimeSeries(batch.getLabels(), output[0]);
            }

            log.info("EPOCH " + i + " TRAIN AUC: " + roc.calculateAUC());

            roc = new ROC(100);
            while (validData.hasNext()) {
                DataSet batch = validData.next();
                INDArray[] output = model.output(batch.getFeatures());
                roc.evalTimeSeries(batch.getLabels(), output[0]);
            }

            log.info("EPOCH " + i + " VALID AUC: " + roc.calculateAUC());

            trainData.reset();
            validData.reset();
            if (i % 5 == 0) {
                log.info("EPOCH" + i + "ModelSerializer Write to File");
                File locationToSave = new File("/tmp/physionet" + i + ".zip");
                ModelSerializer.writeModel(model,locationToSave,true);

            }

        }

        ROC roc = new ROC(100);

        while (testData.hasNext()) {
            DataSet batch = testData.next();
            INDArray[] output = model.output(batch.getFeatures());
            roc.evalTimeSeries(batch.getLabels(), output[0]);
        }
        log.info("***** Test Evaluation *****");
        log.info("{}", roc.calculateAUC());

        // STEP #10 REMOVE THE COMMENT BELOW
        */
    }

}
