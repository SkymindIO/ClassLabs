package ai.skymind.training.solutions;

import org.apache.log4j.BasicConfigurator;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.schema.SequenceSchema;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.eval.ROC;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
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
public class PhysionetMultivariateTimeSeriesClassificationWithOptions {

    // Change directory


    //private static File baseDir = new File("src/main/resources/physionet2012");
    //private static File featuresDir = new File(baseDir, "sequence");

    /* Task-specific configuration */
    //private static File labelsDir = new File(baseDir, "mortality");

    // For logging with SL4J
    private static final Logger log = LoggerFactory.getLogger(PhysionetMultivariateTimeSeriesClassificationWithOptions.class);

    // Number of training, validation, test examples
    public static int NB_INPUTS = 86;
    public static final int NB_TRAIN_EXAMPLES = 3200;
    public static final int NB_VALID_EXAMPLES = 400;
    public static final int NB_TEST_EXAMPLES = 400;

    public static final int NB_EPOCHS = 25;
    public static final int RANDOM_SEED = 1234;
    public static final double LEARNING_RATE = 0.032;
    public static final int BATCH_SIZE = 40;
    public static final int lstmLayerSize = 200;


    public static void main(String[] args) throws IOException, InterruptedException {
        BasicConfigurator.configure();

        File baseDir = new ClassPathResource("physionet2012").getFile();
        File featuresDir = new File(baseDir, "sequence");

    /* Task-specific configuration */
        File labelsDir = new File(baseDir, "mortality");


        // STEP 0: Flags controlling which data

        // 0 for removing Time and Elapsed columns; 1 for removing Time; 2 for removing Elapsed
        int remove = 0;
        int numLabelClasses = 2;
        boolean resampled = false; // If true use resampled data

        DataSetIterator trainData;
        DataSetIterator validData;
        DataSetIterator testData;

        if(resampled){
            NB_INPUTS-=2;
            featuresDir = new File(baseDir, "resampled");

            SequenceRecordReader trainFeatures = new CSVSequenceRecordReader(1, ",");
            trainFeatures.initialize( new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/%d.csv", 0, NB_TRAIN_EXAMPLES - 1));
            SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
            trainLabels.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/%d.csv", 0, NB_TRAIN_EXAMPLES - 1));

            trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels,
                    BATCH_SIZE, numLabelClasses, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

            // Load validation data
            SequenceRecordReader validFeatures = new CSVSequenceRecordReader(1, ",");
            validFeatures.initialize(new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES , NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES  - 1));
            SequenceRecordReader validLabels = new CSVSequenceRecordReader();
            validLabels.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES , NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES  - 1));


            validData = new SequenceRecordReaderDataSetIterator(validFeatures, validLabels,
                    BATCH_SIZE, numLabelClasses, false,SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

            // Load test data
            SequenceRecordReader testFeatures = new CSVSequenceRecordReader(1, ",");
            testFeatures.initialize(new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES+ NB_VALID_EXAMPLES, NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES + NB_TEST_EXAMPLES - 1));
            SequenceRecordReader testLabels = new CSVSequenceRecordReader();
            testLabels.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES+ NB_VALID_EXAMPLES, NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES + NB_TEST_EXAMPLES - 1));


            testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels,
                    BATCH_SIZE, numLabelClasses, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
        }

        else{
            Schema schema =  new SequenceSchema.Builder()
                    .addColumnsDouble("Time","Elapsed","ALP").addColumnCategorical("ALPMissing")
                    .addColumnDouble("ALT").addColumnCategorical("ALTMissing").addColumnDouble("AST")
                    .addColumnCategorical("ASTMissing").addColumnDouble("Age").addColumnCategorical("AgeMissing")
                    .addColumnDouble("Albumin").addColumnCategorical("AlbuminMissing").addColumnDouble("BUN")
                    .addColumnCategorical("BUNMissing").addColumnDouble("Bilirubin").addColumnCategorical("BilirubinMissing")
                    .addColumnDouble("Cholesterol").addColumnCategorical("CholesterolMissing").addColumnDouble("Creatinine")
                    .addColumnCategorical("CreatinineMissing").addColumnDouble("DiasABP").addColumnCategorical("DiasABPMissing")
                    .addColumnDouble("FiO2").addColumnCategorical("FiO2Missing").addColumnDouble("GCS")
                    .addColumnCategorical("GCSMissing").addColumnCategorical("Gender0").addColumnCategorical("Gender1")
                    .addColumnDouble("Glucose").addColumnCategorical("GlucoseMissing").addColumnDouble("HCO3")
                    .addColumnCategorical("HCO3Missing").addColumnDouble("HCT").addColumnCategorical("HCTMissing")
                    .addColumnDouble("HR").addColumnCategorical("HRMissing").addColumnDouble("Height")
                    .addColumnCategorical("HeightMissing").addColumnCategorical("ICUType1").addColumnCategorical("ICUType2")
                    .addColumnCategorical("ICUType3").addColumnCategorical("ICUType4").addColumnDouble("K")
                    .addColumnCategorical("KMissing").addColumnDouble("Lactate").addColumnCategorical("LactateMissing")
                    .addColumnDouble("MAP").addColumnCategorical("MAPMissing").addColumnDouble("MechVent")
                    .addColumnCategorical("MechVentMissing").addColumnDouble("Mg").addColumnCategorical("MgMissing")
                    .addColumnDouble("NIDiasABP").addColumnCategorical("NIDiasABPMissing").addColumnDouble("NIMAP")
                    .addColumnCategorical("NIMAPMissing").addColumnDouble("NISysABP").addColumnCategorical("NISysABPMissing")
                    .addColumnDouble("Na").addColumnCategorical("NaMissing").addColumnDouble("PaCO2")
                    .addColumnCategorical("PaCO2Missing").addColumnDouble("PaO2").addColumnCategorical("PaO2Missing")
                    .addColumnDouble("Platelets").addColumnCategorical("PlateletsMissing").addColumnDouble("RespRate")
                    .addColumnCategorical("RespRateMissing").addColumnDouble("SaO2").addColumnCategorical("SaO2Missing")
                    .addColumnDouble("SysABP").addColumnCategorical("SysABPMissing").addColumnDouble("Temp")
                    .addColumnCategorical("TempMissing").addColumnDouble("TroponinI").addColumnCategorical("TroponinIMissing")
                    .addColumnDouble("TroponinT").addColumnCategorical("TroponinTMissing").addColumnDouble("Urine")
                    .addColumnCategorical("UrineMissing").addColumnDouble("WBC").addColumnCategorical("WBCMissing")
                    .addColumnDouble("Weight").addColumnCategorical("WeightMissing").addColumnDouble("pH")
                    .addColumnCategorical("pHMissing").build();
            
            TransformProcess transformProcess;

            if(remove == 0){
                transformProcess = new TransformProcess.Builder(schema).removeColumns("Time", "Elapsed").build();
                NB_INPUTS-=2;
            }
            else if(remove == 1){
                transformProcess = new TransformProcess.Builder(schema).removeColumns("Time").build();
                NB_INPUTS-=1;
            }
            else if(remove == 2){
                transformProcess = new TransformProcess.Builder(schema).removeColumns("Elapsed").build();
                NB_INPUTS-=1;
            }
            else{
                transformProcess = new TransformProcess.Builder(schema).build();
            }

            // Load training data
            SequenceRecordReader trainFeatures = new CSVSequenceRecordReader(1, ",");
            trainFeatures.initialize( new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/%d.csv", 0, NB_TRAIN_EXAMPLES - 1));
            SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
            trainLabels.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/%d.csv", 0, NB_TRAIN_EXAMPLES - 1));

            TransformProcessSequenceRecordReader trainRemovedFeatures = new TransformProcessSequenceRecordReader(trainFeatures, transformProcess);

            trainData = new SequenceRecordReaderDataSetIterator(trainRemovedFeatures, trainLabels,
                    BATCH_SIZE, numLabelClasses, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

            // Load validation data
            SequenceRecordReader validFeatures = new CSVSequenceRecordReader(1, ",");
            validFeatures.initialize(new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES , NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES  - 1));
            SequenceRecordReader validLabels = new CSVSequenceRecordReader();
            validLabels.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES , NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES  - 1));
            TransformProcessSequenceRecordReader validRemovedFeatures = new TransformProcessSequenceRecordReader(validFeatures, transformProcess);


            validData = new SequenceRecordReaderDataSetIterator(validRemovedFeatures, validLabels,
                    BATCH_SIZE, numLabelClasses, false,SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

            // Load test data
            SequenceRecordReader testFeatures = new CSVSequenceRecordReader(1, ",");
            testFeatures.initialize(new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES+ NB_VALID_EXAMPLES, NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES + NB_TEST_EXAMPLES - 1));
            SequenceRecordReader testLabels = new CSVSequenceRecordReader();
            testLabels.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES+ NB_VALID_EXAMPLES, NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES + NB_TEST_EXAMPLES - 1));
            TransformProcessSequenceRecordReader testRemovedFeatures = new TransformProcessSequenceRecordReader(testFeatures, transformProcess);


            testData = new SequenceRecordReaderDataSetIterator(testRemovedFeatures, testLabels,
                    BATCH_SIZE, numLabelClasses, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
        }

        // STEP 1: ETL/vectorization


        // STEP 2: Model configuration and initialization

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(RANDOM_SEED)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(LEARNING_RATE)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.ADAM)
                .graphBuilder()
                .addInputs("trainFeatures")
                .setOutputs("predictMortality")
                .addLayer("L1", new GravesLSTM.Builder()
                                .nIn(NB_INPUTS)
                                .nOut(lstmLayerSize)
                                .activation(Activation.TANH)
                                .build(),
                        "trainFeatures")
                .addLayer("predictMortality", new RnnOutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .nIn(lstmLayerSize).nOut(numLabelClasses).build(),"L1")
                .pretrain(false).backprop(true)
                .build();

        // STEP 3 Performance monitoring

        ComputationGraph model = new ComputationGraph(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        // STEP 4 Model training

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
    }
}
