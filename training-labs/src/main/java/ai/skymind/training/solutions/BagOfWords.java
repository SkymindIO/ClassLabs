package ai.skymind.training.solutions;

/**
 * Created by tomhanlon on 2/3/17.
 */

import org.apache.log4j.BasicConfigurator;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.bagofwords.vectorizer.BagOfWordsVectorizer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareFileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * Created by tomhanlon on 12/30/16.
 */



public class BagOfWords {
    private static Logger log = LoggerFactory.getLogger(BagOfWords.class);

    public static void main(String[] args) throws Exception{
        BasicConfigurator.configure();
        //reads directory bow, which contains 2 files
        File rootDir = new ClassPathResource("bow").getFile();
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();

        //LabelAwareSentenceIterator iter = new LabelAwareFileSentenceIterator(new File("/Users/tomhanlon/SkyMind/java/Class_Labs/src/main/resources/bow/"));
        LabelAwareSentenceIterator iter = new LabelAwareFileSentenceIterator(new ClassPathResource("bow").getFile());


        List<String> labels = Arrays.asList("label1", "label2");


        BagOfWordsVectorizer vectorizer = new BagOfWordsVectorizer.Builder()
                .setMinWordFrequency(1)
                .setStopWords(new ArrayList<String>())
                .setTokenizerFactory(tokenizerFactory)
                .setIterator(iter)
                .build();


        vectorizer.fit();
        log.info(vectorizer.toString());


        File file = new File("/Users/tomhanlon/SkyMind/java/Class_Labs/src/main/resources/bow/file2.txt");

        log.info("Number of Words in Bag");
        log.info(Integer.toString(vectorizer.getVocabCache().numWords()));


        INDArray array = vectorizer.transform("world world world");
        log.info("Transformed array: world world world " + array);

        INDArray array2 = vectorizer.transform("file");
        log.info("Transformed array: file " + array2);

        INDArray array3 = vectorizer.transform("world file Hello");
        log.info("Transformed array: file " + array3);

        VocabWord word =vectorizer.getVocabCache().wordFor("one.");
        log.info("WORD HERE" + word.getLabel() + word.getVocabId() + word.getIndex());
        log.info(Long.toString(word.getSequencesCount()));
        //DataSet dataSet = vectorizer.vectorize("This is 2 file.", "label2");

        for (int index = 0; index < vectorizer.getVocabCache().numWords(); index++){
            log.info(vectorizer.getVocabCache().wordAtIndex(index));
            log.info(String.valueOf(vectorizer.getVocabCache().docAppearedIn(vectorizer.getVocabCache().wordAtIndex(index))));
        }
        // #### progresss #####
        log.info(vectorizer.getVocabCache().tokens().toString());
        //log.info(vectorizer.getIndex("world"));

        log.info(vectorizer.getVocabCache().wordAtIndex(1));
        //vectorizer.transform("hello");
        log.info(vectorizer.toString());
        //vectorizer.getIndex().toString();
        vectorizer.getVocabCache().wordFrequency("hello");

        log.info("Labels used: " + vectorizer.getVocabCache().totalNumberOfDocs());

    }

}
