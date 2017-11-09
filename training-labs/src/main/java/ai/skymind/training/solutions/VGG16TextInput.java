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

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;

/**
 * Created by tomhanlon on 1/23/17.
 */
public class VGG16TextInput {

    public  static void main(String[] args) throws Exception {
        BasicConfigurator.configure();
        File savedNetwork = new ClassPathResource("vgg16.zip").getFile();

        //File locationToSave = new File("/Users/tomhanlon/SkyMind/java/Class_Labs/vgg16.zip");

        ComputationGraph vgg16 = ModelSerializer.restoreComputationGraph(savedNetwork);

        InputStreamReader r = new InputStreamReader(System.in);
        BufferedReader br = new BufferedReader(r);
        for (; ; ){
            System.out.println("type EXIT to close");
            System.out.println("Enter Image Path to predict with VGG16");
            System.out.print("File Path: ");
            String path = br.readLine();
            if ("EXIT".equals(path))
                break;
            System.out.println("You typed" + path);

            //File file = new File("/Users/tomhanlon/tensorflow/vgg16/keras-model-zoo/deep-learning-models/images/lion.jpeg");
            File file = new File(path);
            NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
            INDArray image = loader.asMatrix(file);
            DataNormalization scaler = new VGG16ImagePreProcessor();
            scaler.transform(image);
            System.out.print(image);
            INDArray[] output = vgg16.output(false,image);
            System.out.println(TrainedModels.VGG16.decodePredictions(output[0]));

        }


    }



}



