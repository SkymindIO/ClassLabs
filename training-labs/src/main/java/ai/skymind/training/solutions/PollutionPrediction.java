package ai.skymind.training.solutions;

import org.apache.log4j.BasicConfigurator;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * Created by tomhanlon on 2/23/18.
 */
public class PollutionPrediction {

    private static final Logger log = LoggerFactory.getLogger(PollutionPrediction.class);

    public static void main(String[] args) throws Exception {
        BasicConfigurator.configure();

        SequenceRecordReader train = new CSVSequenceRecordReader(0, ",");

        /*
        public SequenceRecordReaderDataSetIterator(org.datavec.api.records.reader.SequenceRecordReader reader,
                                           int miniBatchSize,
                                           int numPossibleLabels,
                                           int labelIndex,
                                           boolean regression)
            Constructor where features and labels come from the SAME RecordReader (i.e., target/label is a column in the same data as the features)
            Parameters:
            reader - SequenceRecordReader with data
            miniBatchSize - size of each minibatch
            numPossibleLabels - number of labels/classes for classification (or not used if regression == true)
            labelIndex - index in input of the label index
            regression - Whether output is for regression or classification

         */
        // trainFeatures.initialize(new NumberedFileInputSplit(UCIData.featuresDirTrain.getAbsolutePath() + "/%d.csv", 0, 449));
            //train.initialize(new FileSplit(new File("/Users/tomhanlon/SkyMind/java/new_class_labs_Nov_2017/trainingLabs-parent/training-labs/src/main/resources/Pollution/pollution_train")));

        train.initialize(new FileSplit(new ClassPathResource("Pollution/pollution_trained_transformed").getFile()));
        SequenceRecordReaderDataSetIterator trainData = new SequenceRecordReaderDataSetIterator(train, 2, 1,13,true);

        //System.out.println(train.next().toString());

        INDArray features = trainData.next().getFeatures();
        //INDArray labels = trainData.next().getLabels();

        System.out.println(features.shapeInfoToString());
        // SHAPE SHOULD BE, [samples, timesteps, features].
        // In this case [batch,1,13]
        //System.out.println(features);

        //System.out.println(trainData.getLabels().toString());


    }
}
