package ai.skymind.training.labs;

import org.apache.log4j.BasicConfigurator;
import org.datavec.api.util.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * Created by tomhanlon on 2/23/17.
 */
public class AbeloneFeedForwardNetwork {
    private static Logger log = LoggerFactory.getLogger(AbeloneFeedForwardNetwork.class);
    public static void main(String[] args) throws Exception {
        BasicConfigurator.configure();

        /*
        Lab Intro:
        Welcome to the Lab
        Some parameters are set here in for you in the beginning
         */


        int numLinesToSkip = 0; // The data file has no header
        String delimiter = ","; // The data file is comma delimited
        int batchSize = 600; // Run 600 records at a time before updating weights
        int seed = 123; // Set a random seed for reproducible results
        int labelIndex = 8; // the 9th column is the label, number of rings or age
        int numClasses = 30; // Age 0-29
        int numOutputs = 30; // age 0-29
        double learningRate = 0.005; // How fast to adjust weights
        int numInputs = 8; // 8 params
        int numHiddenNodes = 40; // two hidden layers 40 nodes each
        int nEpochs = 50; // number of total passes through the data
        int iterations = 100; // number of iterations

        /*
        ############# STEP 3  ##############
        Define paths to the training data and test data
         Create File objects traindata and testdata by using DataVecs ClassPathResource
         */

        File traindata = new ClassPathResource("abalone/abalone_train.csv").getFile();
        File testdata = new ClassPathResource("abalone/abalone_test.csv").getFile();


        /*
        ######## Lab Step 4 #########
        Create and initialize a Record Reader for the training data and the testdata
        Datavec JavaDoc is https://deeplearning4j.org/datavecdoc/

        Record Reader JavaDoc is https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/RecordReader.html

        Two lines of code for each
        1: define the RecordReader "Recordeader [rrname] = new CSVRecordReader()
        2: rrname.initialize(new FileSplit(File))

        If you are curious and want to explore RecordReader
        call next and see the List of Writables returned
        attach a Listener
        setListeners
         */


        // Add Your code here 4 lines



        /*
        ######## Lab Step 5 ###########
        Create a DataSetIterator for each
        RecordReaderDataSetIterator prepares the data to feed into our neural network
        https://deeplearning4j.org/doc/org/deeplearning4j/datasets/datavec/RecordReaderDataSetIterator.html
         For the Constructor for RecordReaderDataSetIterator us (RecordReader,batchSize,labelIndex,numClasses)
         Those variables are defined earlier

         */


        // Add your code here, two lines one for train data one for test data

        // DataSetIterator trainIter....
        // DataSetIterator testIter .....



        /*

       #### Lab Step 6 ####
        Build the Neural network
        You rarely start completely from scratch deep learning is driven by what has
        worked in recent papers and competitions.
        For a basic network like this you might start by looking at our examples
        https://github.com/deeplearning4j/dl4j-examples
        The code is build for your, important parameters are left out, fill in the blanks
        OptimizationAlGO see https://deeplearning4j.org/doc/org/deeplearning4j/nn/api/OptimizationAlgorithm.html
        WeightInit see https://deeplearning4j.org/doc/org/deeplearning4j/nn/weights/WeightInit.html
        Activation: For this cluster we have one output node for each class 0-29
        The activation Function SOFTMAX takes the weights of each output node and converts to a probability
        It would also be possible to have a single output node emit a continuous value and use that to get a range 0-29

        */





        /*

                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed) // Consistent seed for consistent results
                .iterations(iterations) // number of parameter update passes
                .optimizationAlgo(OptimizationAlgorithm.****YOUR CODE HERE****)
                .learningRate(learningRate)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .regularization(true).l2(1e-4)
                .weightInit(WeightInit.*** YOUR CODE HERE***)
                .activation(Activation.*** YOUR CODE HERE***)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.*** YOUR CODE HERE****)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build();


        */

        /*


        /*
        ######## LAB STEP 7 #########

        Uncomment the code below to run your model,
        No changes need to be made.
        Make sure the Iterators match the names you gave your training and Testing Iterator
         */


        /*
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));    //Print score every 10 parameter updates

        for ( int n = 0; n < nEpochs; n++) {
            model.fit( trainIter );
        }

        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(numOutputs);
        while(testIter.hasNext()){
            DataSet t = testIter.next();
            INDArray features = t.getFeatureMatrix();
            INDArray lables = t.getLabels();
            INDArray predicted = model.output(features,false);

            eval.eval(lables, predicted);

        }


        System.out.println(eval.stats());


*/


    }
}
