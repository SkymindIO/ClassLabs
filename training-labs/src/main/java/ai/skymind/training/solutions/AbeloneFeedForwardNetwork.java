package ai.skymind.training.solutions;

import org.apache.log4j.BasicConfigurator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
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
        int numLinesToSkip = 0;
        String delimiter = ",";
        int batchSize = 600;
        int seed = 123;
        int labelIndex = 8;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
        int numClasses = 30;
        int numOutputs = 30;
        double learningRate = 0.005;
        int numInputs = 8;
        int numHiddenNodes = 40;
        int nEpochs = 50;
        int iterations = 100;


        File traindata = new ClassPathResource("abalone/abalone_train.csv").getFile();
        File testdata = new ClassPathResource("abalone/abalone_test.csv").getFile();


        //final String filenameTrain  = new org.nd4j.linalg.io.ClassPathResource("/classification/saturn_data_train.csv").getFile().getPath();
        //final String filenameTest  = new org.nd4j.linalg.io.ClassPathResource("/classification/saturn_data_eval.csv").getFile().getPath();

        //Load the training data:
        RecordReader rrtrain = new CSVRecordReader();
        rrtrain.initialize(new FileSplit(traindata));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rrtrain,batchSize,labelIndex,numClasses);
        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(testdata));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,labelIndex,numClasses);


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .regularization(true).l2(1e-4)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.TANH)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build();


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





    }
}
