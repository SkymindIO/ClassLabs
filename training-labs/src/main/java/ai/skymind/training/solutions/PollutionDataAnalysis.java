package ai.skymind.training.solutions;

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
import java.util.List;

/**
 * Created by tomhanlon on 2/10/18.
 */
public class PollutionDataAnalysis {
    public static void main(String[] args) throws Exception {

        // The data analysis file will be written to the /tmp directory
        // In order to allow the code to run more than once
        // without an IO exception
        // the filepath will include the minute of the day
        // the next five lines build a filepath for our output

        Calendar rightNow = Calendar.getInstance();
        int hour = rightNow.get(Calendar.HOUR_OF_DAY);
        int minute = rightNow.get(Calendar.MINUTE);
        int minuteOfDay = ((hour * 60) + minute);
        String outputPath = "/tmp/Pollution_data_" + minuteOfDay;


        // Create LOCAL spark conf
        SparkConf conf = new SparkConf();
        conf.setMaster("local[*]");
        conf.setAppName("Pollution Data");

        JavaSparkContext sc = new JavaSparkContext(conf);

        // Get a path to the original datafile

        String file = new ClassPathResource("Pollution/pollution_test").getFile().getAbsolutePath();
        JavaRDD<String> stringData = sc.textFile(file);


        //We first need to parse this comma-delimited (CSV) format; we can do this using CSVRecordReader:
        RecordReader rr = new CSVRecordReader();
        JavaRDD<List<Writable>> parsedInputData = stringData.map(new StringToWritablesFunction(rr));

        //  count the lines, just for fun
        /*
        System.out.println("##################");
        System.out.print(parsedInputData.rdd().count());
        System.out.println("##################");
        */


        // Build Schema to represent data as stored on disk
        /*


        No: row number
        year: year of data in this row
        month: month of data in this row
        day: day of data in this row
        hour: hour of data in this row
        pm2.5: PM2.5 concentration (ug/m^3)
        DEWP: Dew Point (â„ƒ)
        TEMP: Temperature (â„ƒ)
        PRES: Pressure (hPa)
        cbwd: Combined wind direction
        Iws: Cumulated wind speed (m/s)
        Is: Cumulated hours of snow
        Ir: Cumulated hours of rain
         */


        Schema schema = new Schema.Builder()
                .addColumnsInteger("LineNumber","year","month","day","hour","pm","dewp","temp","pres")
                .addColumnCategorical("winddir","NW","cv","NE","SE")
                .addColumnsDouble("windsp")
                .addColumnsInteger("inchesrain","inchessnow","imputedpm","targetpm")
                .build();

        // Analyze the data
        // This data turns out to be normalized between 0-1 already
       int maxHistogramBuckets = 10;
       DataAnalysis dataAnalysis = AnalyzeSpark.analyze(schema, parsedInputData, maxHistogramBuckets);
       String dataAnalysisString = dataAnalysis.toString();

        //write to html
       HtmlAnalysis.createHtmlAnalysisFile(dataAnalysis, new File(outputPath + "/PollutionAnalysis.html"));
        FileUtils.writeStringToFile(new File(outputPath + "/analysis.txt"),dataAnalysisString);


        TransformProcess tp = new TransformProcess.Builder(schema)
                .removeColumns("LineNumber")
                .categoricalToInteger("winddir")
                .build();

        // Create a new RDD by applying TransformProcess to current RDD
        JavaRDD<List<Writable>> processed = SparkTransformExecutor.execute(parsedInputData,tp);

        // convert Writable back to string for export
        JavaRDD<String> toSave= processed.map(new WritablesToStringFunction(","));

        // Save to a directory in /tmp with the analysis, the original and the processed
        toSave.saveAsTextFile(outputPath + "/processed");
        stringData.saveAsTextFile(outputPath + "/original");
    }
}

