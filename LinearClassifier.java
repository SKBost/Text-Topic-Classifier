import java.util.*; 
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;

public class LinearClassifier {

    private final int numFeatures;
    private final ArrayList<DataPoint> trPts; // training data points with feature vectors and labels

    private static double placeholder = 420.69;

    public LinearClassifier(int newNumFeatures, String trainFile) {

        numFeatures = newNumFeatures;
        print("todo: read in files");
        print("todo: get actual number of points");
        trPts = readDataFrom(trainFile);

        print("todo: finish LinearClassifier constructor");

        

    }

    private static double[] multiplyScalarAndVector(double scalar, double[] vector) {
        //print("inside multiplyScalarAndVector");
        double[] newVec = new double[vector.length];
        for(int i = 0; i < vector.length; i++) {
            newVec[i] = vector[i] * scalar;
        }
        /*
        print("multiplying scalar " + scalar  + "and the following vector:");
        print("");
        printDoubleArray(vector);
        print("multiplied vector is ");
        print("");
        printDoubleArray(newVec);
        */
        return newVec;
    }

    // runs a single pass of perceptron given a starting value
    public double[] teachPerceptron(double[] w0) { // note: we start at w0 instead of w1 for 0 indexing
        print("inside teachPerceptron");
        double[][] w = new double[trPts.size() + 1][numFeatures]; // todo: do we actually use numPoints + 1?
        w[0] = w0;

        for(int t = 0; t < trPts.size(); t++) {
            double yt = trPts.get(t).getLabel();
            double[] xt  = trPts.get(t).getFeatures();
            double classVal = yt * getDotProduct(w[t], xt); // classification value

            if(classVal <= 0) {
                //print("perceptron got it wrong on training point t = " + t + "; updating w[t+1]");
                w[t+1] = sumVectors(w[t], multiplyScalarAndVector(yt, xt));
            } else {
                w[t+1] = w[t];
            }
        }

        return w[trPts.size()]; // final form of w after running on last data point
    }

    private double[] sumVectors(double[] first, double[] second) {
        //print("inside sumVectors");

        double[] newVec = new double[first.length];
        
        if(first.length != second.length) {
            print("ERROR in sumVectors: first.length != second.length; returning placeholder zero vector");
        } else {
            for(int i = 0; i < first.length; i++) {
                newVec[i] = first[i] + second[i];
            }
        }

        return newVec;
    }

    private double getDotProduct(double[] wt, double[] xt) {
        //print("inside getDotProduct");
        // check if inputs are valid
        if(wt.length != xt.length) {
            print("ERROR in getDotProduct: wt.length != xt.length; returning placeholder " + placeholder);
            return placeholder;
        }

        // get inner/dot product
        double multDimSum = 0;
        for(int i = 0; i < wt.length; i++) {
            multDimSum += wt[i] * xt[i];
            //print("multDimSum (now " + multDimSum + ") = previous multDimSum + wt[" + i +"] (" + wt[i] + ") * xt[" + i + "](" + xt[i] + ")");
        }
        return multDimSum;
    }

    /*
    // this method might be trivial
    private double[] getZeroVector() {
        double[] zeroVec = new double[dimNum];
        return zeroVec;
    }
    */


    public void runPerceptronPasses(int numPasses) {
        print("inside runPerceptronPasses");
        double[] w0 = new double[numFeatures]; // starter values of 0 for perceptron to build off
        for(int i = 0; i < numPasses; i++) {
            double[] wT = teachPerceptron(w0); // the final value/output
            double error = getPerceptronTrainingError(wT);
            print("after run " + i + " of teaching perceptron, we get the following training error:" + error);
            //printDoubleArray(w0);
            w0 = wT; // the final value/output of each pass is the starter for the next pass
        }
    }

    public double getPerceptronError(double[] classifier, ArrayList<DataPoint> testData) {
        int timesCorrect = 0;
        int timesIncorrect = 0;

        // for each line of data with a feature vector and label
        for(int i = 0; i < testData.size(); i++) {

            DataPoint testDP = testData.get(i);
            double label = testDP.getLabel();

            // classify based on training data with our perceptron
            if(perceptronGivesCorrectClassification(classifier, testDP)) { // if classification is correct
                timesCorrect++; // increment how many times we got it right
            } else {
                timesIncorrect++;
                //print("for line " + i + " of the test data, we " + 
                 //     " incorrectly label it as " + classification + 
                //     " instead of the correct label, " + label);
                    
            }
            
        }

        double error = ((double) timesIncorrect) / testData.size();
        //print("error  = " + error);

        //print("--------------------------------------------");

        return error;
    }

    // w is the classifier (todo: change name)
    // perceptron gives correct classification
    private boolean perceptronGivesCorrectClassification(double[] w, DataPoint testDP) {
        double yt = testDP.getLabel();
        double[] xt  = testDP.getFeatures();
        double classVal = yt * getDotProduct(w, xt); // classification value
        return !(classVal <= 0);
    }

    public double getPerceptronTrainingError(double[] classifier) {
        return getPerceptronError(classifier, trPts);
    }

    public static void printDoubleArray(double[] array) {
        System.out.print("[");
            for(int j = 0; j < array.length; j++) {
                System.out.print(array[j] + ", ");
            }
            System.out.println("]");
    }

    public ArrayList<DataPoint> readDataFrom(String fileName) {
        return readDataFrom(fileName, numFeatures);
    }

    public static ArrayList<DataPoint> readDataFrom(String fileName, int numFeatures) {

        ArrayList<DataPoint> data = new ArrayList<DataPoint>();
        //print("readDataFrom " + fileName);

        try {
            File file = new File(fileName); 
            Scanner sc = new Scanner(file); 
    
            while (sc.hasNextLine()) { // make each line into a data point
                String line = sc.nextLine();
                String[] featuresAndLabel = line.split(" ");

                if(featuresAndLabel.length != numFeatures + 1) {
                    print("error: numFeatures is incorrect");
                    return data;
                } else {
                    //print("numFeatures is correct for this line");
                }

                double[] features = new double[numFeatures];
                for(int i = 0; i < numFeatures; i++) {
                    features[i] = Double.parseDouble(featuresAndLabel[i]);
                }

                double label = Double.parseDouble(featuresAndLabel[numFeatures]);
                //System.out.print(Integer.toString(label));
                DataPoint point = new DataPoint(features, label);
                data.add(point);

            }

        } catch (FileNotFoundException ex) {
            print("Invalid file path given: " + fileName);
        }

        return data;
    }


    private static void print(String text) {
        System.out.println(text);
    }
}