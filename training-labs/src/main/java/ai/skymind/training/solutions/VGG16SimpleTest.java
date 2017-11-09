package ai.skymind.training.solutions;
import org.apache.log4j.BasicConfigurator;
import org.datavec.api.util.ClassPathResource;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;

import java.io.File;


public class VGG16SimpleTest {

    public static final String IMAGE_DIR = "/Users/tomhanlon/tensorflow/vgg16/keras-model-zoo/deep-learning-models/images";
    public static final File parentDir = new File(IMAGE_DIR);
    public static final int batchSize = 2;

    public static void main(String [] args) throws Exception {
        BasicConfigurator.configure();



        //Dataset iterator using an image record reader
       // ImageRecordReader rr = new ImageRecordReader(224,224,3);
       // rr.initialize(new FileSplit(parentDir));
       // RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(rr,batchSize);
       // iter.setCollectMetaData(true);

        //Attach the VGG16 specific preprocessor to the dataset iterator for the mean shifting required
       // DataSetPreProcessor preProcessor = TrainedModels.VGG16.getPreProcessor();
       // iter.setPreProcessor(preProcessor);

        File savedNetwork = new ClassPathResource("vgg16.zip").getFile();

        //File locationToSave = new File("resources/vgg16.zip");       //Where to save the network. Note: the file is in .zip format - can be opened externally

        //Load the model

        ComputationGraph vgg16 = ModelSerializer.restoreComputationGraph(savedNetwork);
        File elephant = new ClassPathResource("elephant.jpeg").getFile();

        //File file = new File("/Users/tomhanlon/tensorflow/vgg16/keras-model-zoo/deep-learning-models/images/lion.jpeg");
        NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
        //NativeImageLoader loader = new NativeImageLoader(10, 10, 3);
        INDArray image = loader.asMatrix(elephant);
        //System.out.print(image);
        //DataNormalization scaler = new
        //scaler.transform(image);
        DataNormalization scaler = new VGG16ImagePreProcessor();
        scaler.transform(image);
        //System.out.print(image);
        INDArray[] output = vgg16.output(false,image);
        System.out.println(TrainedModels.VGG16.decodePredictions(output[0]));



        /*
        Add something like this to open a buffered reader
        InputStreamReader r = new InputStreamReader(System.in);  
        BufferedReader br = new BufferedReader(r);  
        for (; ; ) { 
        System.out.print("Word: "); 
        String word = br.readLine();  
        if ("EXIT".equals(word))
        break;  

        Collection<String> lst = vec.wordsNearest(word, 20);  

        System.out.println(word + " -> " + lst);         } 

         */


    }
}
