package ai.skymind.training.solutions;

import org.apache.log4j.BasicConfigurator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

//import java.lang.reflect.Method;
//import java.lang.reflect.Type;


/**
 * Built for SkyMind Training class
 */
public class TestDataVec {
    private static Logger log = LoggerFactory.getLogger(TestDataVec.class);
    public static void main(String[] args) throws  Exception{
        BasicConfigurator.configure();

        int numLinesToSkip = 0;
        String delimiter = ",";
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
        System.out.println(recordReader.next().getClass());
        System.out.println(recordReader.getClass());
    //recordReader.next()
    }

    }

