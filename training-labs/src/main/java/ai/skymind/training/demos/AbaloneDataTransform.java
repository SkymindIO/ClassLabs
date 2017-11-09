package ai.skymind.training.demos;

import org.apache.commons.io.FileUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.ui.HtmlAnalysis;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.AnalyzeSpark;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.datavec.spark.transform.misc.WritablesToStringFunction;

import java.io.File;
import java.util.Calendar;
import java.util.Date;
import java.util.List;

/**
 * This class demonstrates the analysis and transformation of raw data
 * using datavec transform process and analysis
 * TO USE:
 * Run the code and look for the directory /tmp/abalone_data_xxxx
 * the minute of the day is appended so the code can run more than once without error
 *
 * In that directory
 * analysis.txt // min/max stdev and all that show the data is normalized already
 * "Viscera weight"|Double|DoubleAnalysis(min=5.0E-4,max=0.76,mean=0.18059360785252573,
 *
 * less original/part-00000
 * M,0.455,0.365,0.095,0.514,0.2245,0.101,0.15,15
 * M,0.35,0.265,0.09,0.2255,0.0995,0.0485,0.07,7
 *
 * less processed/part-00000
 * 0,0.455,0.365,0.095,0.514,0.2245,0.101,0.15,15
 * 0,0.35,0.265,0.09,0.2255,0.0995,0.0485,0.07,7
*/

 public class AbaloneDataTransform {
    public static void main(String[] args) throws Exception {


        Calendar rightNow = Calendar.getInstance();
        int hour = rightNow.get(Calendar.HOUR_OF_DAY);
        int minute = rightNow.get(Calendar.MINUTE);
        int minuteOfDay = ((hour * 60) + minute);
        String outputPath = "/tmp/abalone_data_" + minuteOfDay;



        /*
        Data SUMMARY
        Sex		nominal			M, F, and I (infant)
	    Length		continuous	mm	Longest shell measurement
	    Diameter	continuous	mm	perpendicular to length
	    Height		continuous	mm	with meat in shell
	    Whole weight	continuous	grams	whole abalone
	    Shucked weight	continuous	grams	weight of meat
	    Viscera weight	continuous	grams	gut weight (after bleeding)
	    Shell weight	continuous	grams	after being dried
	    Rings		integer			+1.5 gives the age in years

         */

        // Build Schema to represent data as stored on disk
        Schema schema = new Schema.Builder()
                .addColumnCategorical("Sex","M","F","I")
                .addColumnsDouble("Length", "Diameter", "Height", "Whole weight","Shucked weight","Viscera weight","Shell weight")
                .addColumnInteger("Rings")
                .build();


        // Create LOCAL spark conf
        SparkConf conf = new SparkConf();
        conf.setMaster("local[*]");
        conf.setAppName("Abalone Data");

        JavaSparkContext sc = new JavaSparkContext(conf);

        // Create Spark RDD from the data file
        // This is just the lines, they are parsed to writable records in next step
        String file = new ClassPathResource("abalone/abalone.data").getFile().getAbsolutePath();
        JavaRDD<String> stringData = sc.textFile(file);

        //We first need to parse this comma-delimited (CSV) format; we can do this using CSVRecordReader:
        RecordReader rr = new CSVRecordReader();
        JavaRDD<List<Writable>> parsedInputData = stringData.map(new StringToWritablesFunction(rr));

        // Analyze the data
        // This data turns out to be normalized between 0-1 already
        int maxHistogramBuckets = 10;
        DataAnalysis dataAnalysis = AnalyzeSpark.analyze(schema, parsedInputData, maxHistogramBuckets);
        String dataAnalysisString = dataAnalysis.toString();
        // write to html
        HtmlAnalysis.createHtmlAnalysisFile(dataAnalysis, new File(outputPath + "/AbeloneAnalysis.html"));



        // Define a transform process
        // In this case the M/F/I need to be converted to numeric
        // More options if needed are available in TransformProcess
        TransformProcess tp = new TransformProcess.Builder(schema)
                .categoricalToInteger("Sex")
                .build();

        // Create a new RDD by applying TransformProcess to current RDD
        JavaRDD<List<Writable>> processed = SparkTransformExecutor.execute(parsedInputData,tp);

        // convert Writable back to string for export
        JavaRDD<String> toSave= processed.map(new WritablesToStringFunction(","));

        // Save to a directory in /tmp with the analysis, the original and the processed
        toSave.saveAsTextFile(outputPath + "/processed");
        stringData.saveAsTextFile(outputPath + "/original");
        FileUtils.writeStringToFile(new File(outputPath + "/analysis.txt"),dataAnalysisString);



    }
}
