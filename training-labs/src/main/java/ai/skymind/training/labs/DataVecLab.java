package ai.skymind.training.labs;

import org.apache.log4j.BasicConfigurator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class DataVecLab {

    private static Logger log = LoggerFactory.getLogger(DataVecLab.class);

    public static void main(String[] args) throws  Exception {
        BasicConfigurator.configure();
/*
        //Set the parameters numLinesToSkip and Delimiter
        // #### YOUR CODE HERE ####


        // Create and initialize a RecordReader passing it a file object
         // #### YOUR CODE HERE ####



        // Set params for RecordReaderDataSetIterator
        // labelIndex
        // numClasses
        // batchSize

        // #### YOUR CODE HERE ####


        // Add your DataSetIterator here

        // #### YOUR CODE HERE ####

        // Create a DataSet called allData by calling the next() method on the iterator

        // #### YOUR CODE HERE ####

        // Call the shuffle method on the allData DataSet to shuffle the records

        // #### YOUR CODE HERE ####

        // Split the data into Test and Train

        // #### YOUR CODE HERE ####





        // Code below is a working Neural Net for this dataset


        final int numInputs = 4;
        int outputNum = 3;
        int iterations = 1000;
        long seed = 6;


        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .activation(Activation.TANH)
            .weightInit(WeightInit.XAVIER)
            .learningRate(0.1)
            .regularization(true).l2(1e-4)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3)
                .build())
            .layer(1, new DenseLayer.Builder().nIn(3).nOut(3)
                .build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX)
                .nIn(3).nOut(outputNum).build())
            .backprop(true).pretrain(false)
            .build();

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        model.fit(trainingData);
        System.out.println(model.conf().toJson());

        //evaluate the model on the test set
        Evaluation eval = new Evaluation(3);
        INDArray output = model.output(testData.getFeatureMatrix());
        System.out.println(output);
        eval.eval(testData.getLabels(), output);
        log.info(eval.stats());


*/
    }

}

