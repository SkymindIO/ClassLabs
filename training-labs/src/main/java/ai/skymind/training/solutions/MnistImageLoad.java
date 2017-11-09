package ai.skymind.training.solutions;

import org.apache.log4j.BasicConfigurator;
import org.datavec.api.util.ClassPathResource;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * /**
 *  This code example is featured in this youtube video
 *
 *  https://www.youtube.com/watch?v=zrTSs715Ylo
 *
 ** This differs slightly from the Video Example,
 * The Video example had the data already downloaded
 * This example includes code that downloads the data
 *
 * Data SOurce
 *  wget http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz
 *  followed by tar xzvf mnist_png.tar.gz
 *
 *  OR
 *  git clone https://github.com/myleott/mnist_png.git
 *  cd mnist_png
 *  tar xvf mnist_png.tar.gz
 *
 *
 *
 *  This examples builds on the MnistImagePipelineExample
 *  by Saving the Trained Network
 *
 */
public class MnistImageLoad {
    /** Data URL for downloading */

    private static Logger log = LoggerFactory.getLogger(MnistImageLoad.class);

    public static void main(String[] args) throws Exception {
        BasicConfigurator.configure();
        // image information
        // 28 * 28 grayscale
        // grayscale implies single channel
        int height = 28;
        int width = 28;
        int channels = 1;


        File modelSave =  new ClassPathResource("trained_mnist_model.zip").getFile();
        File imageToTest = new ClassPathResource("dog.jpeg").getFile();
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelSave);


        // Use NativeImageLoader to convert to numerical matrix

        NativeImageLoader loader = new NativeImageLoader(height, width, channels);

        // Get the image into an INDarray

        INDArray image = loader.asMatrix(imageToTest);

        // 0-255
        // 0-1
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.transform(image);
        // Pass through to neural Net

        INDArray output = model.output(image);

        log.info(output.toString());




    }


}
