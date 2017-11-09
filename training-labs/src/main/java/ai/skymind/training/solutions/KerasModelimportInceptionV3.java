package ai.skymind.training.solutions;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;


/**
 * Created by tomhanlon on 4/20/17.
 */
public class KerasModelimportInceptionV3 {
    public static void main(String[] args) throws Exception{


        //Path to Saved Model and weights
        // We use DataVec's ClassPathResource here, you could pass the import files path string directly

        String kerasModelfromKerasExport = new ClassPathResource("inception_v3_complete.h5").getFile().getPath();

        /*
        Create a MultiLayerNetwork from the saved model
         */

        // MultiLayerNetwork model = org.deeplearning4j.nn.modelimport.keras.KerasModelImport.importKerasSequentialModelAndWeights(kerasModelfromKerasExport);

        // ComputationGraph model = KerasModelImport.importKerasModelAndWeights(kerasModelfromKerasExport);
        //ComputationGraph model = KerasModelImport.importKerasModelAndWeights(kerasModelfromKerasExport);
        ComputationGraph model = KerasModelImport.importKerasModelAndWeights("/Users/tomhanlon/tensorflow/vgg16/keras-model-zoo/deep-learning-models/inception_V3_config","/Users/tomhanlon/tensorflow/vgg16/keras-model-zoo/deep-learning-models/inception_v3.h5",false);
        //ComputationGraph model = new ComputationGraph()

       // File elephant = new ClassPathResource("elephant.jpeg").getFile();

        //File file = new File("/Users/tomhanlon/tensorflow/vgg16/keras-model-zoo/deep-learning-models/images/lion.jpeg");
        //NativeImageLoader loader = new NativeImageLoader(299, 299, 3);
        //NativeImageLoader loader = new NativeImageLoader(10, 10, 3);
        //INDArray image = loader.asMatrix(elephant);
        //System.out.print(image);
        //DataNormalization scaler = new
        //scaler.transform(image);
        //DataNormalization scaler = new VGG16ImagePreProcessor();
        //scaler.transform(image);
        //System.out.print(image);
        //INDArray[] output = model.output(false,image);
        //System.out.println(TrainedModels.VGG16.decodePredictions(output[0]));

        //System.out.println(output);

    }

}

