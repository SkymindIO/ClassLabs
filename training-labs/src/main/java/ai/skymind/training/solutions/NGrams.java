package ai.skymind.training.solutions;

import org.apache.log4j.BasicConfigurator;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.NGramTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;


/**
 * Created by tomhanlon on 12/30/16.
 */



public class NGrams {
    private static Logger log = LoggerFactory.getLogger(NGrams.class);

    public NGrams() {
    }

    public static void main(String[] args) throws Exception{
        BasicConfigurator.configure();
        String toTokenize = "To boldly go where no one has gone before.";
        TokenizerFactory factory = new NGramTokenizerFactory(new DefaultTokenizerFactory(), 1, 2);
        Tokenizer tokenizer = factory.create(toTokenize);
        factory = new NGramTokenizerFactory(new DefaultTokenizerFactory(), 2, 3);
        List<String> tokens = factory.create(toTokenize).getTokens();
        log.info(tokens.toString());


    }

}
