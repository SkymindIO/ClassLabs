package ai.skymind.training.solutions;

import org.apache.log4j.BasicConfigurator;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Random;

//import spire.math.algebraic.Mul;

/*GravesLSTM Character modelling example


   Example: Train a LSTM RNN to generates text, one character at a time.
	This example is somewhat inspired by Andrej Karpathy's blog post,
	"The Unreasonable Effectiveness of Recurrent Neural Networks"
	http://karpathy.github.io/2015/05/21/rnn-effectiveness/

	You may find similar examples that train on the wikipedia corpus,
	the google news corpus, the linux kernel source code,
	or the complete works of Shakespeare

	This example uses weather forecasts archived from the
	US National Weather service office in Charleson West Virginia

	Weather forecast products have a particular text data format,
	also some weather terms are abbreviated.

	Note that the model trains in a semi-supervised way, it predicts the next character
	and training occurs based on how correct the prediction is.

	The model is then asked to generate text on it's own provided with a seed of text.

	What does the model end up "knowing"?
	It gets words sorted out fairly quickly.
	Long term dependencies, like Mon before tues, before wed. That takes longer.

	You could modify this and feed it specific strings to probe what it can do.

    For more details on RNNs in DL4J, see the following:
    http://deeplearning4j.org/usingrnns
    http://deeplearning4j.org/lstm
    http://deeplearning4j.org/recurrentnetwork
 */
public class  GravesLSTMCharModellingWeatherForecasts {
	public static void main( String[] args ) throws Exception {
		BasicConfigurator.configure();
		int lstmLayerSize = 200;					//Number of units in each GravesLSTM layer
		int miniBatchSize = 32;						//Size of mini batch to use when  training
		int exampleLength = 4000;					//Length of each training example sequence to use. This could certainly be increased
        int tbpttLength = 50;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
		int numEpochs = 1;							//Total number of training epochs
        int generateSamplesEveryNMinibatches = 10;  //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
		int nSamplesToGenerate = 4;					//Number of samples to generate after each training epoch
		int nCharactersToSample = 1200;				//Length of each sample to generate
		// String generationInitialization = null;		//Optional character initialization; a random character is used if null
		String generationInitialization = "WVZ006-171700-\n" +
				"CABELL-\n" +
				"INCLUDING THE CITY OF...HUNTINGTON\n" +
				"932 PM EST FRI JUN 16 2016";
		// Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
		// Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default
		Random rng = new Random(12345);

		//Get a DataSetIterator that handles vectorization of text into something we can use to train
		// our GravesLSTM network.
		CharacterIterator iter = getWeatherReportIterator(miniBatchSize,exampleLength);
		int nOut = iter.totalOutcomes();


		/*
		#### LAB STEP 1 #####
		Begin by uncommenting the lines below and then add to the Builder
		the settings described in the instructions.


		 */

		//MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
		//		.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)



		//Set up network configuration:
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
			.learningRate(0.1)
			.seed(12345)
			.regularization(true)
			.l2(0.001)
            .weightInit(WeightInit.XAVIER)
            .updater(new RmsProp(0.95))
			.list()
			.layer(0, new GravesLSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
					.activation(Activation.TANH).build())
			.layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
					.activation(Activation.TANH).build())
			.layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation(Activation.SOFTMAX)        //MCXENT + softmax for classification
					.nIn(lstmLayerSize).nOut(nOut).build())
            .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
			.pretrain(false).backprop(true)
			.build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(1));

		//Print the  number of parameters in the network (and for each layer)
		Layer[] layers = net.getLayers();
		int totalNumParams = 0;
		for( int i=0; i<layers.length; i++ ){
			int nParams = layers[i].numParams();
			System.out.println("Number of parameters in layer " + i + ": " + nParams);
			totalNumParams += nParams;
		}
		System.out.println("Total number of network parameters: " + totalNumParams);

		//Do training, and then generate and print samples from network
        int miniBatchNumber = 0;
		for( int i=0; i<numEpochs; i++ ){
            while(iter.hasNext()){
                DataSet ds = iter.next();
                net.fit(ds);
                if(++miniBatchNumber % generateSamplesEveryNMinibatches == 0){
                    System.out.println("--------------------");
                    System.out.println("Completed " + miniBatchNumber + " minibatches of size " + miniBatchSize + "x" + exampleLength + " characters" );
                    System.out.println("Sampling characters from network given initialization \"" + (generationInitialization == null ? "" : generationInitialization) + "\"");
                    String[] samples = sampleCharactersFromNetwork(generationInitialization,net,iter,rng,nCharactersToSample,nSamplesToGenerate);
                    for( int j=0; j<samples.length; j++ ){
                        System.out.println("----- Sample " + j + " -----");
                        System.out.println(samples[j]);
                        System.out.println();
                    }
                }
            }

			iter.reset();	//Reset iterator for another epoch
		}
		File locationToSave = new File("trained_graves_weather.zip");

		// boolean save Updater
		boolean saveUpdater = false;

		// ModelSerializer needs modelname, saveUpdater, Location

		ModelSerializer.writeModel(net,locationToSave,saveUpdater);

		System.out.println("\n\nExample complete");
	}

	/** Downloads Shakespeare training data and stores it locally (temp directory). Then set up and return a simple
	 * DataSetIterator that does vectorization based on the text.
	 * @param miniBatchSize Number of text segments in each training mini-batch
	 * @param sequenceLength Number of characters in each text segment.
	 */
	public static CharacterIterator getWeatherReportIterator(int miniBatchSize, int sequenceLength) throws Exception{
		//The Complete Works of William Shakespeare
		//5.3MB file in UTF-8 Encoding, ~5.4 million characters
		//https://www.gutenberg.org/ebooks/100
		File f = new ClassPathResource("weather.txt").getFile();
		String fileLocation = f.getAbsolutePath();
		//String fileLocation = tempDir + "/Shakespeare.txt";	//Storage location from downloaded file
		//File f = new File(fileLocation);

		if(!f.exists()) throw new IOException("File does not exist: ");	//Download problem?

		char[] validCharacters = CharacterIterator.getMinimalCharacterSet();	//Which characters are allowed? Others will be removed
		return new CharacterIterator(fileLocation, Charset.forName("UTF-8"),
				miniBatchSize, sequenceLength, validCharacters, new Random(12345));
	}

	/** Generate a sample from the network, given an (optional, possibly null) initialization. Initialization
	 * can be used to 'prime' the RNN with a sequence you want to extend/continue.<br>
	 * Note that the initalization is used for all samples
	 * @param initialization String, may be null. If null, select a random character as initialization for all samples
	 * @param charactersToSample Number of characters to sample from network (excluding initialization)
	 * @param net MultiLayerNetwork with one or more GravesLSTM/RNN layers and a softmax output layer
	 * @param iter CharacterIterator. Used for going from indexes back to characters
	 */
	private static String[] sampleCharactersFromNetwork(String initialization, MultiLayerNetwork net,
                                                        CharacterIterator iter, Random rng, int charactersToSample, int numSamples ){
		//Set up initialization. If no initialization: use a random character
		if( initialization == null ){
			initialization = String.valueOf(iter.getRandomCharacter());
		}

		//Create input for initialization
		INDArray initializationInput = Nd4j.zeros(numSamples, iter.inputColumns(), initialization.length());
		char[] init = initialization.toCharArray();
		for( int i=0; i<init.length; i++ ){
			int idx = iter.convertCharacterToIndex(init[i]);
			for( int j=0; j<numSamples; j++ ){
				initializationInput.putScalar(new int[]{j,idx,i}, 1.0f);
			}
		}

		StringBuilder[] sb = new StringBuilder[numSamples];
		for( int i=0; i<numSamples; i++ ) sb[i] = new StringBuilder(initialization);

		//Sample from network (and feed samples back into input) one character at a time (for all samples)
		//Sampling is done in parallel here
		net.rnnClearPreviousState();
		INDArray output = net.rnnTimeStep(initializationInput);
		output = output.tensorAlongDimension(output.size(2)-1,1,0);	//Gets the last time step output

		for( int i=0; i<charactersToSample; i++ ){
			//Set up next input (single time step) by sampling from previous output
			INDArray nextInput = Nd4j.zeros(numSamples,iter.inputColumns());
			//Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
			for( int s=0; s<numSamples; s++ ){
				double[] outputProbDistribution = new double[iter.totalOutcomes()];
				for( int j=0; j<outputProbDistribution.length; j++ ) outputProbDistribution[j] = output.getDouble(s,j);
				int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution,rng);

				nextInput.putScalar(new int[]{s,sampledCharacterIdx}, 1.0f);		//Prepare next time step input
				sb[s].append(iter.convertIndexToCharacter(sampledCharacterIdx));	//Add sampled character to StringBuilder (human readable output)
			}

			output = net.rnnTimeStep(nextInput);	//Do one time step of forward pass
		}

		String[] out = new String[numSamples];
		for( int i=0; i<numSamples; i++ ) out[i] = sb[i].toString();
		return out;
	}

	/** Given a probability distribution over discrete classes, sample from the distribution
	 * and return the generated class index.
	 * @param distribution Probability distribution over classes. Must sum to 1.0
	 */
	public static int sampleFromDistribution( double[] distribution, Random rng ){
		double d = rng.nextDouble();
		double sum = 0.0;
		for( int i=0; i<distribution.length; i++ ){
			sum += distribution[i];
			if( d <= sum ) return i;
		}
		//Should never happen if distribution is a valid probability distribution
		throw new IllegalArgumentException("Distribution is invalid? d="+d+", sum="+sum);
	}
}
