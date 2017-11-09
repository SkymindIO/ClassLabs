package ai.skymind.training.solutions;

import org.apache.log4j.BasicConfigurator;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.bagofwords.vectorizer.BagOfWordsVectorizer;
import org.deeplearning4j.text.documentiterator.FilenamesLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;


/**
 * Created by tomhanlon on 12/30/16.
 */



public class BagofWords2 {
    private static Logger log = LoggerFactory.getLogger(BagofWords2.class);

    public static void main(String[] args) throws Exception{
        BasicConfigurator.configure();
        /*
        Read a Directory of files
        Label them with the filename
         */
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();

        LabelAwareIterator iterator = new FilenamesLabelAwareIterator.Builder()
                .addSourceFolder(new ClassPathResource("bow").getFile())
                .useAbsolutePathAsLabel(false)
                .build();



        while(iterator.hasNext()){
            LabelledDocument doc = iterator.nextDocument();
            System.out.println(doc.getContent());
            System.out.println(doc.getLabels().get(0));
        }

        iterator.reset();

        //BagOfWordsVectorizer vectorizer2 = new BagOfWordsVectorizer.Builder()

        BagOfWordsVectorizer vectorizer = new BagOfWordsVectorizer.Builder()
                .setMinWordFrequency(1)
                .setStopWords(new ArrayList<String>())
                .setTokenizerFactory(tokenizerFactory)
                .setIterator(iterator)
                .build();


        vectorizer.fit();
        //DataSet dataSet = vectorizer.vectorize("This is 2 file.", "label2");

        // private double tfForWord(String word, int documentLength) {
       // return MathUtils.tf(vocabCache.wordFrequency(word), documentLength);
        //}


        log.info(vectorizer.getVocabCache().tokens().toString());
        System.out.println(vectorizer.getVocabCache().totalNumberOfDocs());
        System.out.println(vectorizer.getVocabCache().docAppearedIn("two."));
        System.out.println(vectorizer.getVocabCache().docAppearedIn("one."));
        System.out.println(vectorizer.getVocabCache().docAppearedIn("world"));
        //System.out.println(vectorizer.getIndex().document(1));

    }

}
